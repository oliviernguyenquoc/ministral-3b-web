import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
import torch

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Ministral-3B-Instruct")  # adapte si besoin
TRAIN_FILE = os.getenv("TRAIN_FILE", "train.jsonl")
EVAL_FILE = os.getenv("EVAL_FILE", "eval.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./lora-privacy")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "./lora-privacy-adapter")

# Bigger-machine defaults (override via env vars)
LORA_R = int(os.getenv("LORA_R", "32"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
PER_DEVICE_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM", "4"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
LEARNING_RATE = float(os.getenv("LR", "2e-4"))
NUM_EPOCHS = int(os.getenv("EPOCHS", "2"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "400"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "400"))
GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "true").lower() == "true"
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# Load datasets
train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
eval_ds = load_dataset("json", data_files=EVAL_FILE, split="train")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# QLoRA-friendly LoRA config
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    fp16=not USE_BF16,
    bf16=USE_BF16,
    logging_steps=25,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    evaluation_strategy="steps",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    report_to="none",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=LOAD_IN_4BIT,
    torch_dtype=TORCH_DTYPE,
    device_map="auto",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Saved adapter to {ADAPTER_DIR}")
