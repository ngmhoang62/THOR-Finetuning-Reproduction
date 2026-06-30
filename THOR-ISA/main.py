import argparse
import yaml
import torch
from attrdict import AttrDict
import pandas as pd

from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer


class ThorRunner:
    def __init__(self, args):
        config = AttrDict(yaml.safe_load(open(args.config, 'r', encoding='utf-8')))
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)

        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        self.config = config

    def forward(self):
        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        else:
            raise 'Should choose a correct reasoning mode: prompt or thor.'

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return

        print("Fine-tuning mode for training.")
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'thor'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=True,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-d', '--data_name', default='laptops', choices=['restaurants', 'laptops'],
                        help='semeval data name')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    
    # Extensions CLI args overrides
    parser.add_argument('--use_lora', type=str, default=None, choices=['True', 'False'], help='Force enable/disable LoRA')
    parser.add_argument('--self_consistency', type=str, default=None, choices=['True', 'False'], help='Force enable/disable Self Consistency')
    parser.add_argument('--self_consistency_n', type=int, default=None, help='Number of sampling paths')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace Token for Gemma')
    parser.add_argument('--model_path', type=str, default=None, help='Override base model path')
    parser.add_argument('--model_size', type=str, default=None, help='Override base model size label')

    args = parser.parse_args()
    
    # Process string booleans and set overrides
    runner = ThorRunner(args)
    if args.use_lora is not None:
        runner.config.use_lora = args.use_lora == 'True'
    if args.self_consistency is not None:
        runner.config.self_consistency = args.self_consistency == 'True'
    if args.self_consistency_n is not None:
        runner.config.self_consistency_n = args.self_consistency_n
    if args.hf_token is not None:
        runner.config.hf_token = args.hf_token
    if args.model_path is not None:
        runner.config.model_path = args.model_path
    if args.model_size is not None:
        runner.config.model_size = args.model_size
        
    runner.forward()

