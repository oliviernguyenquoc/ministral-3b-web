import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Ministral-3B-Instruct")
TRAIN_FILE = os.getenv("TRAIN_FILE", "train_v2.jsonl")
EVAL_FILE = os.getenv("EVAL_FILE", "eval_v2.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./lora-privacy")
ADAPTER_DIR = os.getenv("ADAPTER_DIR", "./lora-privacy-adapter")
LOGGING_DIR = os.getenv("LOGGING_DIR", "./runs")
REPORT_TO = os.getenv("REPORT_TO", "tensorboard")
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT", None)

# LoRA params
LORA_R = int(os.getenv("LORA_R", "32"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TARGET_MODULES = os.getenv("TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj").split(",")

# Train params
PER_DEVICE_BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM", "4"))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "2048"))
LEARNING_RATE = float(os.getenv("LR", "2e-4"))
NUM_EPOCHS = int(os.getenv("EPOCHS", "2"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "400"))
EVAL_STEPS = int(os.getenv("EVAL_STEPS", "400"))
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS", "25"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.03"))
PACKING = os.getenv("PACKING", "false").lower() == "true"
SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "2"))
LOAD_BEST_MODEL_AT_END = os.getenv("LOAD_BEST_MODEL_AT_END", "true").lower() == "true"

# Precision / quant
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
GRADIENT_CHECKPOINTING = os.getenv("GRADIENT_CHECKPOINTING", "true").lower() == "true"

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
TORCH_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def save_metrics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    eval_ds = load_dataset("json", data_files=EVAL_FILE, split="train")
    if "text" not in train_ds.column_names or "text" not in eval_ds.column_names:
        raise ValueError("Both train and eval datasets must include a 'text' field.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=not USE_BF16,
        bf16=USE_BF16,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        report_to=[x.strip() for x in REPORT_TO.split(",")] if REPORT_TO != "none" else [],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        torch_dtype=TORCH_DTYPE,
        device_map="auto",
    )

    if GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=PACKING,
    )

    print("[CONFIG]")
    print(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "train_file": TRAIN_FILE,
                "eval_file": EVAL_FILE,
                "output_dir": OUTPUT_DIR,
                "adapter_dir": ADAPTER_DIR,
                "logging_dir": LOGGING_DIR,
                "report_to": REPORT_TO,
                "bf16": USE_BF16,
                "load_in_4bit": LOAD_IN_4BIT,
                "batch_size": PER_DEVICE_BATCH_SIZE,
                "grad_accum": GRAD_ACCUM_STEPS,
                "epochs": NUM_EPOCHS,
                "lr": LEARNING_RATE,
                "save_steps": SAVE_STEPS,
                "eval_steps": EVAL_STEPS,
            },
            indent=2,
        )
    )

    train_result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    eval_metrics = trainer.evaluate()

    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_metrics(
        Path(OUTPUT_DIR) / "training_metrics.json",
        {
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "global_step": trainer.state.global_step,
        },
    )

    print(f"[DONE] adapter saved to {ADAPTER_DIR}")
    print(f"[DONE] training metrics saved to {Path(OUTPUT_DIR) / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
