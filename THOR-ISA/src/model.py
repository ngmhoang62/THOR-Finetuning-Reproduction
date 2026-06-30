import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM


class LLMBackbone(nn.Module):
    def __init__(self, config):
        super(LLMBackbone, self).__init__()
        self.config = config
        
        # Check if loading a gated/specific model that requires a token
        kwargs = {}
        if hasattr(config, 'hf_token') and config.hf_token and config.hf_token != 'XXX':
            kwargs['token'] = config.hf_token
            
        model_path = config.model_path.lower()
        self.is_causal = 't5' not in model_path
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Backbone Model
        if self.is_causal:
            # Gemma/Causal LM loading. We enforce text-only by freezing any vision parameters if they exist
            self.engine = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.bfloat16, device_map='auto', **kwargs)
            
            # Disable vision model to save VRAM and computation
            for name, param in self.engine.named_parameters():
                if 'vision' in name or 'multi_modal' in name or 'img' in name or 'visual' in name:
                    param.requires_grad = False
        else:
            self.engine = T5ForConditionalGeneration.from_pretrained(config.model_path, **kwargs)

        # Apply LoRA if configured
        if hasattr(config, 'use_lora') and config.use_lora:
            from peft import LoraConfig, get_peft_model
            
            target_modules = ["q", "v"] if not self.is_causal else ["q_proj", "v_proj"]
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM" if self.is_causal else "SEQ_2_SEQ_LM"
            )
            self.engine = get_peft_model(self.engine, peft_config)
            print("LoRA Adapter successfully injected.")

    def forward(self, **kwargs):
        if not self.is_causal:
            input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
            input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
            output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
            output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                                 decoder_attention_mask=output_masks, labels=output_ids)
            loss = output[0]
            return loss
        else:
            # For Causal LMs, we concatenate input and target for causal fine-tuning
            input_ids, input_masks, output_ids = [kwargs[w] for w in '\
            input_ids, input_masks, output_ids'.strip().split(', ')]
            
            batch_size = input_ids.shape[0]
            # Replace padding in targets
            labels_list = []
            full_input_ids_list = []
            full_attention_mask_list = []
            
            for i in range(batch_size):
                inp = input_ids[i][input_masks[i] == 1]
                out = output_ids[i][output_ids[i] != self.tokenizer.pad_token_id]
                
                full = torch.cat([inp, out], dim=0)
                # Labels for causal lm: ignore input indices (-100), predict only outputs
                label = torch.cat([torch.full_like(inp, -100), out], dim=0)
                
                full_input_ids_list.append(full)
                full_attention_mask_list.append(torch.ones_like(full))
                labels_list.append(label)
                
            # Pad sequences to create batch
            full_input_ids = torch.nn.utils.rnn.pad_sequence(full_input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            full_attention_mask = torch.nn.utils.rnn.pad_sequence(full_attention_mask_list, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
            
            full_input_ids = full_input_ids.to(self.config.device)
            full_attention_mask = full_attention_mask.to(self.config.device)
            labels = labels.to(self.config.device)
            
            output = self.engine(input_ids=full_input_ids, attention_mask=full_attention_mask, labels=labels)
            return output.loss

    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        
        # Support Self-Consistency parameters dynamically if set
        gen_kwargs = {
            'max_length': self.config.max_length,
            'input_ids': input_ids,
            'attention_mask': input_masks
        }
        if hasattr(self.config, 'self_consistency') and self.config.self_consistency:
            gen_kwargs['num_return_sequences'] = self.config.self_consistency_n
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = 0.7
            gen_kwargs['top_p'] = 0.9
            
        output = self.engine.generate(**gen_kwargs)
        
        # Postprocess decoded outputs
        if self.is_causal:
            # Strip input prompt length to get generated response only
            prompt_lens = [len(self.tokenizer.decode(ids, skip_special_tokens=True)) for ids in input_ids]
            dec = [self.tokenizer.decode(ids) for ids in output]
            
            cleaned_outputs = []
            n_seq = self.config.self_consistency_n if (hasattr(self.config, 'self_consistency') and self.config.self_consistency) else 1
            for idx, text in enumerate(dec):
                batch_idx = idx // n_seq
                # Extract only Assistant's text or substring after the prompt length
                original_prompt = self.tokenizer.decode(input_ids[batch_idx])
                clean_text = text.replace(original_prompt, '').replace('<pad>', '').replace('</s>', '').replace('<bos>', '').replace('<eos>', '').strip()
                # Clean specific Gemma tag residues if present
                clean_text = clean_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
                cleaned_outputs.append(clean_text)
            return cleaned_outputs
        else:
            dec = [self.tokenizer.decode(ids) for ids in output]
            output = [context.replace('<pad>', '').replace('</s>', '').replace('<bos>', '').replace('<eos>', '').strip() for context in dec]
            return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=self.config.max_length)
        
        dec = []
        if self.is_causal:
            prompt_lens = [len(self.tokenizer.decode(ids, skip_special_tokens=True)) for ids in input_ids]
            for idx, ids in enumerate(output):
                original_prompt = self.tokenizer.decode(input_ids[idx])
                text = self.tokenizer.decode(ids)
                clean_text = text.replace(original_prompt, '').replace('<pad>', '').replace('</s>', '').replace('<bos>', '').replace('<eos>', '').strip()
                clean_text = clean_text.split('<|im_start|>assistant')[-1].split('<|im_end|>')[0].strip()
                dec.append(clean_text)
        else:
            dec = [self.tokenizer.decode(ids).replace('<pad>', '').replace('</s>', '').replace('<bos>', '').replace('<eos>', '').strip() for ids in output]
            
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.lower().strip(), 0) for w in dec]
        return output

