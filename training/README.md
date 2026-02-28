# Privacy Risk Analyzer — Training Pipeline (V1 + V2)

This repo trains a small local model (e.g. ~3B) to analyze conversation exports and return **strict JSON** risk scores for 5 categories:

- `health_vulnerability`
- `personal_relationships_conflicts`
- `financial_vulnerability`
- `risky_confessions_secrets`
- `location_routine_tracking`

The pipeline is intentionally split into two versions:

- **V1 (Dataset-membership supervision)**: no weak-labeling. We treat dataset origin as ground-truth category to teach *format + categorization*.
- **V2 (Teacher scoring)**: a larger model produces calibrated 0–100 scores + evidence, then we fine-tune the student to match it.

---

## 0) Repo Structure

Recommended files:

├── build_v1_dataset.py # prepares train.jsonl / eval.jsonl (V1 labels from dataset membership)
├── train_qlora.py # QLoRA/LoRA training (student)
├── evaluate.py # basic eval (JSON validity, sanity checks)
├── teacher_label.py # (V2) label conversations with a large teacher model
├── train_teacher_sft.py # (V2) SFT on teacher-labeled dataset (optional separate file)
└── README.md

> You can start with V1 only, then add V2 once you have teacher infrastructure.

---

## 1) Datasets Used

### Health
- https://huggingface.co/datasets/solomonk/reddit_mental_health_posts
- https://huggingface.co/datasets/UCSD26/medical_dialog

### Relationships / Family
- https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments (filter subreddits: `relationships`, `Parenting`, `family`)

### Secrets / Confessions
- https://huggingface.co/datasets/SocialGrep/one-million-reddit-confessions
- https://huggingface.co/datasets/shahules786/prosocial-confessions
- https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments (filter: `tifu` / `todayifuckedup`)

### Finance
- https://huggingface.co/datasets/Akhil-Theerthala/Personal-Finance-Queries
- https://huggingface.co/datasets/Akhil-Theerthala/Kuvera-PersonalFinance-V2.1

---

## 2) Environment Setup

### Python dependencies
```bash
pip install -U "datasets>=2.18" transformers peft trl bitsandbytes accelerate sentencepiece