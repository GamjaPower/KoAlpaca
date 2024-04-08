# Works on A100 80G single GPU
python run_clm.py \
--model_name_or_path='beomi/llama-2-ko-7b' \
--train_file='./KoAlpaca_v1.1a_textonly_edited.json' \
--num_train_epochs=3 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--fp16 \
--output_dir='llama-2-ko-7b-v1.1b' \
--do_train \
--optim='adafactor' \
--learning_rate='2e-5' \
--logging_strategy='steps' \
--logging_first_step \
--run_name='llama-2-ko-7b-v1.1b-singlegpu' \
--low_cpu_mem_usage
