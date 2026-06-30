# Hướng dẫn tái lập thực nghiệm THOR

File markdown này hướng dẫn cách tái lập lại thực nghiệm trong bài báo THOR (Reasoning Implicit Sentiment with Chain-of-Thought Prompting).

**Thông tin nhóm thực hiện (4 thành viên):**
- Nguyễn Minh Hoàng - 23520530
- Lưu Gia Huy - 23520618
- Nguyễn Phong Huy - 23520637
- Phạm Hải Đăng - 23520233

**Môn học:** Xử lý ngôn ngữ tự nhiên nâng cao
**Giảng viên:** TS. Nguyễn Thị Quý

## Cấu trúc thư mục

```text
├── THOR-ISA/ # Thư mục chứa toàn bộ mã nguồn chính của phương pháp
│   ├── config/ # Chứa các file cấu hình như config.yaml định nghĩa tham số
│   ├── data/ # Thư mục lưu trữ dữ liệu (Laptops, Restaurants)
│   ├── src/ # Chứa mã nguồn cốt lõi (loader.py, model.py, engine.py)
│   ├── main.py # File script thực thi chính
│   └── requirements.txt # Danh sách các thư viện cần thiết
├── THOR_ISA_Colab.ipynb # Notebook Jupyter dùng để setup môi trường và chạy tự động
└── Reasoning Implicit Sentiment with Chain-of-Thought Prompting.pdf # File paper gốc
```

## Các thay đổi chính so với mã nguồn gốc

Mã nguồn ban đầu đã được bổ sung và tinh chỉnh lại cho đầy đủ với mô tả trong paper cũng như thử nghiệm cấu hình mới:
- **Tích hợp thêm cơ chế Self-Consistency**: Mã nguồn ban đầu bị thiếu phần này, nay đã được bổ sung vào `src/model.py` để sinh nhiều chuỗi kết quả và lấy đa số, giúp tăng cường khả năng suy luận.
- **Áp dụng LoRA**: Tích hợp LoRA cho cấu hình Flan-T5-XXL + THOR do thiếu thốn tài nguyên để full fine-tune mô hình có kích thước lớn.
- **Thử nghiệm mô hình Gemma-3 (4B)**: Bổ sung luồng xử lý và tham số cho các mô hình Causal LM thế hệ mới như Gemma-3-4B-It (thay vì chỉ dùng T5 Encoder-Decoder). Mã nguồn cũng tự động vô hiệu hóa các layer vision của Gemma-3 để tối ưu hóa VRAM.

## Hướng dẫn từng bước chạy thực nghiệm

**Bước 1: Clone kho lưu trữ và thiết lập môi trường**
Tải mã nguồn về và cài đặt các thư viện cần thiết. Nếu bạn chạy trên Colab, bước này cũng tương đương với việc mount Google Drive:
```bash
git clone https://github.com/ngmhoang62/THOR-Finetuning-Reproduction.git
cd THOR-Finetuning-Reproduction/THOR-ISA
```

**Bước 2: Cấu hình Hugging Face Token (dành cho mô hình gated)**
Để tải các mô hình yêu cầu cấp quyền như Gemma-3, bạn cần có Hugging Face Token:
- **Trong file cấu hình**: Mở file `config/config.yaml` và cập nhật trường `hf_token: 'token_của_bạn'`.
- **Trong file notebook**: Cập nhật biến `HF_TOKEN = "token_của_bạn"` tại cell thực thi mô hình Gemma-3.

**Bước 3: Cấu hình tham số LoRA và Self-Consistency**
Các tham số LoRA đã được định nghĩa sẵn bên trong `src/model.py` khi truyền cờ kích hoạt `--use_lora True`, với các giá trị cụ thể như sau:
- `r = 16`
- `lora_alpha = 32`
- `target_modules = ["q", "v"]` (cho Flan-T5) hoặc `["q_proj", "v_proj"]` (cho Gemma-3)
- `lora_dropout = 0.05`
- `bias = "none"`

Cơ chế Self-Consistency được điều khiển thông qua cấu hình trong quá trình sinh (`generate`), ví dụ như sử dụng `do_sample=True`, `temperature=0.7`, `top_p=0.9`.

**Bước 4: Chạy các thực nghiệm cụ thể**
Các câu lệnh để chạy cụ thể từng thực nghiệm (Flan-T5-base, Flan-T5-XXL + LoRA, Gemma-3-4B-It) cũng như các thực nghiệm ablation (bỏ Self-Consistency, bỏ Reason-Revising) **đã được viết sẵn đầy đủ và chi tiết thành từng cell trong file notebook `THOR_ISA_Colab.ipynb`**. Bạn chỉ việc mở notebook này trên Colab (yêu cầu GPU A100) và chạy lần lượt từ trên xuống dưới.

*Lưu ý cấu hình bổ sung (chưa có trong file notebook):*
- Trong trường hợp muốn thay đổi tham số `batch_size` hoặc `learning_rate` khi chạy các lệnh `main.py`, bạn có thể override bằng cách bổ sung thêm cờ `--batch_size <số>` hoặc `--learning_rate <số>` trực tiếp vào lệnh gọi ở terminal/Colab cell.
- Bạn có thể chỉnh lại giá trị `--seed` (mặc định là 42) nếu muốn đánh giá độ ổn định của các bước sinh ngẫu nhiên trong Self-Consistency.

## Acknowledgments & License

- Đồ án môn học này được thực hiện nhằm mục đích tái hiện (reproduce) lại kết quả của nghiên cứu từ paper: **[Reasoning Implicit Sentiment with Chain-of-Thought Prompting](https://arxiv.org/abs/2305.11255)**.
- Mã nguồn của dự án được kế thừa và phát triển dựa trên repo gốc của tác giả: **[scofield7419/THOR-ISA](https://github.com/scofield7419/thor-isa)**.
- Toàn bộ mã nguồn gốc tuân thủ theo giấy phép **Apache License 2.0**.