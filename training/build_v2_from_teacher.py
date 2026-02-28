import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


CATEGORIES = [
    "health_vulnerability",
    "personal_relationships_conflicts",
    "financial_vulnerability",
    "risky_confessions_secrets",
    "location_routine_tracking",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build train_v2/eval_v2 from teacher-labeled JSONL.")
    p.add_argument("--in", dest="input_path", default="teacher_labeled_10k.jsonl")
    p.add_argument("--train-out", default="train_v2.jsonl")
    p.add_argument("--eval-out", default="eval_v2.jsonl")
    p.add_argument("--prompt-file", default="../src/lib/analysis_prompt.txt")
    p.add_argument("--eval-ratio", type=float, default=0.02)
    p.add_argument("--min-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def safe_json_loads(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return None


def validate_teacher_label(label: Dict) -> bool:
    if not isinstance(label, dict):
        return False
    if "overall_score" not in label or "verdict" not in label or "categories" not in label:
        return False
    cats = label.get("categories")
    if not isinstance(cats, dict):
        return False
    for cat in CATEGORIES:
        if cat not in cats or not isinstance(cats[cat], dict):
            return False
        if "score" not in cats[cat] or "risk" not in cats[cat]:
            return False
        ev = cats[cat].get("evidence", [])
        if not isinstance(ev, list):
            return False
    return True


def make_text(prompt: str, conversation: str, label: Dict) -> str:
    return (
        prompt
        + "\n\nConversation:\n"
        + conversation
        + "\n\nJSON:\n"
        + json.dumps(label, ensure_ascii=False)
    )


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    in_path = Path(args.input_path)
    train_out = Path(args.train_out)
    eval_out = Path(args.eval_out)
    prompt = load_prompt(Path(args.prompt_file))

    if not in_path.exists():
        raise FileNotFoundError(f"Teacher file not found: {in_path}")

    rows = []
    total = 0
    invalid = 0
    unlabeled = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total += 1
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)

            conversation = ex.get("conversation")
            if not isinstance(conversation, str) or not conversation.strip():
                invalid += 1
                continue

            label = safe_json_loads(ex.get("teacher_label"))
            if label is None:
                unlabeled += 1
                continue
            if not validate_teacher_label(label):
                print(f"[WARN] line={line_no} invalid teacher label schema")
                invalid += 1
                continue

            row = {
                "id": ex.get("id", f"v2-{line_no:06d}"),
                "source": ex.get("source", ex.get("meta", {}).get("v1_source", "unknown")),
                "bucket": ex.get("bucket", "unknown"),
                "conversation": conversation.strip(),
                "label_json": json.dumps(label, ensure_ascii=False),
                "teacher_label": label,
                "text": make_text(prompt, conversation.strip(), label),
            }
            rows.append(row)

    if not rows:
        raise RuntimeError("No valid labeled rows were found.")

    rng.shuffle(rows)
    n_eval = max(args.min_eval, int(len(rows) * args.eval_ratio))
    n_eval = min(n_eval, len(rows) - 1)
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)

    print(f"[DONE] input_total={total} valid={len(rows)} unlabeled={unlabeled} invalid={invalid}")
    print(f"[DONE] train={len(train_rows)} eval={len(eval_rows)}")
    print(f"[DONE] wrote: {train_out} and {eval_out}")


if __name__ == "__main__":
    main()
