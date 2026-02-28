import os
import json
import re
from typing import Dict, Any, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Ministral-3B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./lora-privacy-adapter")
EVAL_FILE = os.getenv("EVAL_FILE", "eval.jsonl")

def extract_first_json(s: str) -> Dict[str, Any] | None:
    # Find a JSON object in the output
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

@torch.inference_mode()
def generate(model, tok, prompt: str, max_new_tokens: int = 350) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)

def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tok

def load_finetuned():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, tok

def main():
    ds = load_dataset("json", data_files=EVAL_FILE, split="train")

    base_model, base_tok = load_base()
    ft_model, ft_tok = load_finetuned()

    stats = {
        "base": {"json_ok": 0, "n": 0, "fp_high_on_neg": 0, "neg_n": 0},
        "ft": {"json_ok": 0, "n": 0, "fp_high_on_neg": 0, "neg_n": 0},
    }

    for ex in ds.select(range(min(500, len(ds)))):  # limit for speed
        prompt = ex["text"].split("\n\nJSON:\n")[0] + "\n\nJSON:\n"  # only prompt part
        is_neg = str(ex.get("source","")).startswith("negatives")

        # baseline
        b_out = generate(base_model, base_tok, prompt)
        b_json = extract_first_json(b_out)
        stats["base"]["n"] += 1
        if b_json is not None:
            stats["base"]["json_ok"] += 1
            if is_neg:
                stats["base"]["neg_n"] += 1
                if int(b_json.get("overall_score", 0)) >= 50:
                    stats["base"]["fp_high_on_neg"] += 1
        else:
            if is_neg:
                stats["base"]["neg_n"] += 1

        # finetuned
        f_out = generate(ft_model, ft_tok, prompt)
        f_json = extract_first_json(f_out)
        stats["ft"]["n"] += 1
        if f_json is not None:
            stats["ft"]["json_ok"] += 1
            if is_neg:
                stats["ft"]["neg_n"] += 1
                if int(f_json.get("overall_score", 0)) >= 50:
                    stats["ft"]["fp_high_on_neg"] += 1
        else:
            if is_neg:
                stats["ft"]["neg_n"] += 1

    def rate(a,b): return 0 if b==0 else a/b

    print("=== EVAL (up to 500 examples) ===")
    for k in ["base","ft"]:
        print(f"\n[{k}]")
        print("JSON validity:", rate(stats[k]["json_ok"], stats[k]["n"]))
        print("Negatives FP (overall_score>=50):", rate(stats[k]["fp_high_on_neg"], stats[k]["neg_n"]))
        print("Counts:", stats[k])

if __name__ == "__main__":
    main()