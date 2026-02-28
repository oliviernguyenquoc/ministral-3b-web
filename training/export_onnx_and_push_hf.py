import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge LoRA, export ONNX, and optionally upload to Hugging Face Hub.")
    p.add_argument("--base-model", default="mistralai/Ministral-3B-Instruct")
    p.add_argument("--adapter-dir", default="./lora-privacy-adapter")
    p.add_argument("--merged-dir", default="./merged-model")
    p.add_argument("--onnx-dir", default="./onnx-export")
    p.add_argument("--task", default="text-generation-with-past")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--skip-merge", action="store_true")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--push-hf", action="store_true")
    p.add_argument("--hf-repo-id", default="", help="Example: username/ministral-privacy-onnx")
    p.add_argument("--hf-private", action="store_true")
    p.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    p.add_argument("--upload-adapter", action="store_true", help="Also upload the LoRA adapter folder to the same repo.")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def merge_adapter(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tok.save_pretrained(str(merged_dir))
    print(f"[DONE] merged model saved to {merged_dir}")


def export_onnx(merged_dir: Path, onnx_dir: Path, task: str, opset: int) -> None:
    onnx_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        str(merged_dir),
        "--task",
        task,
        "--opset",
        str(opset),
        str(onnx_dir),
    ]
    run_cmd(cmd)
    print(f"[DONE] onnx export saved to {onnx_dir}")


def push_to_hf(repo_id: str, folder: Path, token: str, private: bool, path_in_repo: str = "") -> None:
    if not repo_id:
        raise ValueError("--hf-repo-id is required when --push-hf is used.")
    if not token:
        raise ValueError("HF token is required. Set --hf-token or HF_TOKEN env variable.")
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found for upload: {folder}")

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        path_in_repo=path_in_repo or None,
        commit_message=f"Upload {folder.name}",
    )
    print(f"[DONE] uploaded {folder} to https://huggingface.co/{repo_id}")


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    merged_dir = Path(args.merged_dir)
    onnx_dir = Path(args.onnx_dir)

    if not args.skip_merge:
        merge_adapter(args.base_model, adapter_dir, merged_dir)
    else:
        if not merged_dir.exists():
            raise FileNotFoundError(f"--skip-merge used but merged dir does not exist: {merged_dir}")

    if not args.skip_export:
        export_onnx(merged_dir, onnx_dir, args.task, args.opset)
    else:
        if not onnx_dir.exists():
            raise FileNotFoundError(f"--skip-export used but onnx dir does not exist: {onnx_dir}")

    if args.push_hf:
        push_to_hf(
            repo_id=args.hf_repo_id,
            folder=onnx_dir,
            token=args.hf_token,
            private=args.hf_private,
            path_in_repo="onnx",
        )
        if args.upload_adapter:
            push_to_hf(
                repo_id=args.hf_repo_id,
                folder=adapter_dir,
                token=args.hf_token,
                private=args.hf_private,
                path_in_repo="adapter",
            )

    print("[DONE] export pipeline complete")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as err:
        print(f"[ERROR] command failed with exit code {err.returncode}")
        sys.exit(err.returncode)
