from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Qwen 2.5 7B - 4-bit QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = 2048,
    load_in_4bit = True,
    dtype = "auto",
    device_map = "auto",
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
)

dataset = load_dataset("json", data_files="../dataset/full_training_data.jsonl", split="train")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100,
        learning_rate = 2e-4,
        fp16 = True,
        bf16 = False,
        logging_steps = 5,
        output_dir = "../../models/qwen2.5-7b-finetuned",
        report_to = "none",
        save_strategy = "steps",
        save_steps = 50,
    ),
)
trainer.train()
model.save_pretrained("../../models/qwen2.5-7b-finetuned")
tokenizer.save_pretrained("../../models/qwen2.5-7b-finetuned")
print("[SELESAI] Model Qwen 2.5 7B fine-tuned tersimpan ke models/qwen2.5-7b-finetuned")