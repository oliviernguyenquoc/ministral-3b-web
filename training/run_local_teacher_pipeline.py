import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-command local pipeline: V1 -> candidates -> teacher -> V2.")
    p.add_argument("--build-v1", action="store_true", help="Run build_v1_dataset.py first.")
    p.add_argument("--v1-train", default="train.jsonl")
    p.add_argument("--v1-eval", default="eval.jsonl")
    p.add_argument("--candidates", default="candidates_10k.jsonl")
    p.add_argument("--teacher-out", default="teacher_labeled_10k.jsonl")
    p.add_argument("--train-v2", default="train_v2.jsonl")
    p.add_argument("--eval-v2", default="eval_v2.jsonl")
    p.add_argument("--prompt-file", default="../src/lib/analysis_prompt.txt")
    p.add_argument("--teacher-model", default=os.getenv("TEACHER_MODEL", "mistral-medium-latest"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run(cmd: list[str], env: dict | None = None) -> None:
    print("[RUN]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    py = sys.executable
    env = os.environ.copy()
    env["PROMPT_FILE"] = args.prompt_file
    env["TEACHER_MODEL"] = args.teacher_model

    if args.build_v1:
        run([py, "build_v1_dataset.py"], env=env)

    if not Path(args.v1_train).exists():
        raise FileNotFoundError(
            f"Missing V1 train file: {args.v1_train}. "
            "Run with --build-v1 or provide an existing train file."
        )

    run(
        [
            py,
            "build_candidates_10k.py",
            "--in",
            args.v1_train,
            "--out",
            args.candidates,
            "--seed",
            str(args.seed),
        ],
        env=env,
    )

    if "MISTRAL_API_KEY" not in env:
        raise EnvironmentError("MISTRAL_API_KEY is required for teacher labeling.")

    run(
        [
            py,
            "label_with_mistral_teacher.py",
            "--in",
            args.candidates,
            "--out",
            args.teacher_out,
            "--prompt-file",
            args.prompt_file,
        ],
        env=env,
    )

    run(
        [
            py,
            "build_v2_from_teacher.py",
            "--in",
            args.teacher_out,
            "--train-out",
            args.train_v2,
            "--eval-out",
            args.eval_v2,
            "--prompt-file",
            args.prompt_file,
            "--seed",
            str(args.seed),
        ],
        env=env,
    )

    print("[DONE] local teacher pipeline completed")
    print(f"       candidates: {args.candidates}")
    print(f"       teacher:    {args.teacher_out}")
    print(f"       train_v2:   {args.train_v2}")
    print(f"       eval_v2:    {args.eval_v2}")


if __name__ == "__main__":
    main()
