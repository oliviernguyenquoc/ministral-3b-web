import argparse
import json
import os
import time
from pathlib import Path
from typing import Literal

from mistralai import Mistral
from pydantic import BaseModel, Field, conint


MODEL = os.getenv("TEACHER_MODEL", "mistral-medium-latest")
API_KEY = os.getenv("MISTRAL_API_KEY")


Risk = Literal["none", "low", "medium", "high"]
Verdict = Literal["low", "medium", "high", "critical"]


class CatOut(BaseModel):
    score: conint(ge=0, le=100)
    risk: Risk
    evidence: list[str] = Field(default_factory=list, max_length=3)


class CategoriesOut(BaseModel):
    health_vulnerability: CatOut
    personal_relationships_conflicts: CatOut
    financial_vulnerability: CatOut
    risky_confessions_secrets: CatOut
    location_routine_tracking: CatOut


class OutputSchema(BaseModel):
    verdict: Verdict
    overall_score: conint(ge=0, le=100)
    summary: str
    categories: CategoriesOut


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label candidates with a Mistral teacher model.")
    p.add_argument("--in", dest="input_path", default="candidates_10k.jsonl")
    p.add_argument("--out", dest="output_path", default="teacher_labeled_10k.jsonl")
    p.add_argument("--prompt-file", default="../src/lib/analysis_prompt.txt")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--max-conv-chars", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--resume", action="store_true", help="Resume from existing output file.")
    return p.parse_args()


def load_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return p.read_text(encoding="utf-8").strip()


def build_user_prompt(base_prompt: str, conversation: str) -> str:
    return f"{base_prompt}\n\nConversation:\n{conversation}\n\nJSON:\n"


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def postprocess(label: OutputSchema) -> dict:
    out = label.model_dump()
    for cat_name, cat in out["categories"].items():
        cat["evidence"] = (cat.get("evidence") or [])[:3]
        score = int(cat.get("score", 0))
        if score <= 0:
            cat["risk"] = "none"
            cat["evidence"] = []
    return out


def main() -> None:
    args = parse_args()
    if not API_KEY:
        raise EnvironmentError("MISTRAL_API_KEY is required.")

    in_path = Path(args.input_path)
    out_path = Path(args.output_path)
    prompt = load_prompt(args.prompt_file)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    client = Mistral(api_key=API_KEY)

    skip_lines = 0
    write_mode = "w"
    if args.resume and out_path.exists():
        skip_lines = count_lines(out_path)
        write_mode = "a"
        print(f"[INFO] resume enabled, skipping first {skip_lines} lines")

    written = 0
    failed = 0
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open(write_mode, encoding="utf-8") as f_out:
        for line_no, line in enumerate(f_in, start=1):
            if line_no <= skip_lines:
                continue
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)
            conv = str(ex.get("conversation", ""))[: args.max_conv_chars]
            if not conv.strip():
                ex["teacher_label"] = None
                ex["teacher_error"] = "empty_conversation"
                f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                failed += 1
                continue

            messages = [
                {"role": "system", "content": "Follow the user instruction strictly and return only valid JSON."},
                {"role": "user", "content": build_user_prompt(prompt, conv)},
            ]

            last_error = None
            for attempt in range(args.max_retries):
                try:
                    resp = client.chat.parse(
                        model=args.model,
                        messages=messages,
                        response_format=OutputSchema,
                        temperature=args.temperature,
                    )
                    parsed = resp.choices[0].message.parsed
                    out_json = postprocess(parsed)

                    ex["teacher_label"] = out_json
                    ex["teacher_model"] = args.model
                    ex["teacher_prompt_file"] = args.prompt_file
                    f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    written += 1
                    break
                except Exception as err:  # network and parsing errors
                    last_error = f"{type(err).__name__}: {err}"
                    wait_s = 1.5 * (attempt + 1)
                    print(
                        f"[WARN] line {line_no} attempt {attempt + 1}/{args.max_retries} failed: {last_error} "
                        f"(sleep {wait_s:.1f}s)"
                    )
                    time.sleep(wait_s)
            else:
                ex["teacher_label"] = None
                ex["teacher_error"] = last_error or "unknown_error"
                f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                failed += 1
                print(f"[FAIL] line {line_no}: {ex['teacher_error']}")

            if (written + failed) % 100 == 0:
                f_out.flush()
                print(f"[INFO] progress processed={written + failed} success={written} failed={failed}")

    print(f"[DONE] wrote={written} failed={failed} output={out_path}")


if __name__ == "__main__":
    main()
