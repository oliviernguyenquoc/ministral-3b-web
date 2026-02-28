import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


CATEGORIES = [
    "health_vulnerability",
    "personal_relationships_conflicts",
    "financial_vulnerability",
    "risky_confessions_secrets",
    "location_routine_tracking",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LoRA model quality on JSON outputs.")
    p.add_argument("--model-name", default="mistralai/Ministral-3B-Instruct")
    p.add_argument("--adapter-path", default="./lora-privacy-adapter")
    p.add_argument("--eval-file", default="eval_v2.jsonl")
    p.add_argument("--max-samples", type=int, default=300)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=350)
    p.add_argument("--score-threshold", type=int, default=25, help="Threshold for low vs not-low.")
    p.add_argument("--fp-threshold", type=int, default=50, help="Negative false positive threshold.")
    p.add_argument("--compare-base", action="store_true", help="Also evaluate base model (without adapter).")
    p.add_argument("--report-file", default="evaluation_report.json")
    return p.parse_args()


def extract_first_json(s: str) -> Optional[Dict[str, Any]]:
    # Tries fast regex first, then bracket matching fallback.
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                snippet = s[start : i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


def to_int_score(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_target_label(ex: dict) -> Optional[dict]:
    if isinstance(ex.get("teacher_label"), dict):
        return ex["teacher_label"]
    if isinstance(ex.get("teacher_label"), str):
        try:
            return json.loads(ex["teacher_label"])
        except Exception:
            return None
    if isinstance(ex.get("label_json"), str):
        try:
            return json.loads(ex["label_json"])
        except Exception:
            return None
    return None


def build_prompt_from_text(text: str) -> str:
    if "\n\nJSON:\n" in text:
        return text.split("\n\nJSON:\n", 1)[0] + "\n\nJSON:\n"
    return text


@torch.inference_mode()
def generate(model, tok, prompt: str, max_input_tokens: int, max_new_tokens: int) -> str:
    device = model.device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0], skip_special_tokens=True)


def load_model(model_name: str, adapter_path: Optional[str]):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    if adapter_path:
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
    model.eval()
    return model, tok


def init_stats() -> dict:
    return {
        "n": 0,
        "json_ok": 0,
        "json_invalid": 0,
        "target_available": 0,
        "verdict_accuracy_hits": 0,
        "low_notlow_accuracy_hits": 0,
        "overall_abs_err_sum": 0.0,
        "overall_abs_err_count": 0,
        "per_category_abs_err_sum": {c: 0.0 for c in CATEGORIES},
        "per_category_abs_err_count": {c: 0 for c in CATEGORIES},
        "neg_n": 0,
        "neg_fp_high": 0,
    }


def summarize(stats: dict) -> dict:
    def rate(a: int, b: int) -> float:
        return 0.0 if b == 0 else a / b

    per_cat_mae = {}
    for c in CATEGORIES:
        cnt = stats["per_category_abs_err_count"][c]
        per_cat_mae[c] = 0.0 if cnt == 0 else stats["per_category_abs_err_sum"][c] / cnt

    return {
        "n": stats["n"],
        "json_valid_rate": rate(stats["json_ok"], stats["n"]),
        "json_invalid_count": stats["json_invalid"],
        "target_coverage": rate(stats["target_available"], stats["n"]),
        "verdict_accuracy": rate(stats["verdict_accuracy_hits"], stats["target_available"]),
        "low_vs_not_low_accuracy": rate(stats["low_notlow_accuracy_hits"], stats["target_available"]),
        "overall_score_mae": (
            0.0
            if stats["overall_abs_err_count"] == 0
            else stats["overall_abs_err_sum"] / stats["overall_abs_err_count"]
        ),
        "per_category_mae": per_cat_mae,
        "neg_fp_rate_score_ge_50": rate(stats["neg_fp_high"], stats["neg_n"]),
        "raw": stats,
    }


def evaluate_one_model(
    model_label: str,
    model_name: str,
    adapter_path: Optional[str],
    examples: List[dict],
    max_input_tokens: int,
    max_new_tokens: int,
    score_threshold: int,
    fp_threshold: int,
) -> dict:
    print(f"[LOAD] {model_label} model (adapter={adapter_path or 'none'})")
    model, tok = load_model(model_name, adapter_path)
    stats = init_stats()

    for idx, ex in enumerate(examples, start=1):
        prompt = build_prompt_from_text(ex.get("text", ""))
        target = safe_target_label(ex)
        is_neg = str(ex.get("source", "")).startswith("ultrachat_negative") or str(ex.get("bucket", "")).startswith(
            "negative"
        )

        out_text = generate(model, tok, prompt, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens)
        pred = extract_first_json(out_text)

        stats["n"] += 1
        if pred is None:
            stats["json_invalid"] += 1
            if is_neg:
                stats["neg_n"] += 1
            continue
        stats["json_ok"] += 1

        pred_score = to_int_score(pred.get("overall_score", 0), 0)
        if is_neg:
            stats["neg_n"] += 1
            if pred_score >= fp_threshold:
                stats["neg_fp_high"] += 1

        if target is None:
            continue

        stats["target_available"] += 1
        target_score = to_int_score(target.get("overall_score", 0), 0)
        stats["overall_abs_err_sum"] += abs(pred_score - target_score)
        stats["overall_abs_err_count"] += 1

        if str(pred.get("verdict", "")) == str(target.get("verdict", "")):
            stats["verdict_accuracy_hits"] += 1

        pred_not_low = pred_score >= score_threshold
        target_not_low = target_score >= score_threshold
        if pred_not_low == target_not_low:
            stats["low_notlow_accuracy_hits"] += 1

        pred_cats = pred.get("categories", {})
        tgt_cats = target.get("categories", {})
        if isinstance(pred_cats, dict) and isinstance(tgt_cats, dict):
            for cat in CATEGORIES:
                if cat in pred_cats and cat in tgt_cats:
                    p = to_int_score(pred_cats[cat].get("score", 0), 0)
                    t = to_int_score(tgt_cats[cat].get("score", 0), 0)
                    stats["per_category_abs_err_sum"][cat] += abs(p - t)
                    stats["per_category_abs_err_count"][cat] += 1

        if idx % 25 == 0:
            print(f"[PROGRESS] {model_label}: {idx}/{len(examples)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summarize(stats)


def main() -> None:
    args = parse_args()
    ds = load_dataset("json", data_files=args.eval_file, split="train")
    n = min(args.max_samples, len(ds))
    examples = [ds[i] for i in range(n)]
    print(f"[INFO] loaded {n} examples from {args.eval_file}")

    report: Dict[str, Any] = {}
    report["finetuned"] = evaluate_one_model(
        model_label="finetuned",
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        examples=examples,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        score_threshold=args.score_threshold,
        fp_threshold=args.fp_threshold,
    )

    if args.compare_base:
        report["base"] = evaluate_one_model(
            model_label="base",
            model_name=args.model_name,
            adapter_path=None,
            examples=examples,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            score_threshold=args.score_threshold,
            fp_threshold=args.fp_threshold,
        )
        if report["base"]["json_valid_rate"] > 0:
            report["delta_vs_base"] = {
                "json_valid_rate": report["finetuned"]["json_valid_rate"] - report["base"]["json_valid_rate"],
                "low_vs_not_low_accuracy": report["finetuned"]["low_vs_not_low_accuracy"]
                - report["base"]["low_vs_not_low_accuracy"],
                "overall_score_mae": report["finetuned"]["overall_score_mae"] - report["base"]["overall_score_mae"],
                "neg_fp_rate_score_ge_50": report["finetuned"]["neg_fp_rate_score_ge_50"]
                - report["base"]["neg_fp_rate_score_ge_50"],
            }

    report_path = Path(args.report_file)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[DONE] report saved to {report_path}")


if __name__ == "__main__":
    main()
