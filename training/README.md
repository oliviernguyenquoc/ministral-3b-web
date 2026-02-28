# Pipeline entraînement (Mac -> Brev -> ONNX/HF)

Ce dossier contient une pipeline simple et exécutable pour:
1. Construire V1 (`train.jsonl`, `eval.jsonl`)
2. Construire un paquet `candidates_10k.jsonl`
3. Labeliser au teacher Mistral
4. Convertir en `train_v2.jsonl`, `eval_v2.jsonl`
5. Fine-tuner en QLoRA sur Brev
6. Évaluer correctement
7. Exporter en ONNX et publier sur Hugging Face

Le prompt de référence de l'app est utilisé partout via `src/lib/analysis_prompt.txt`.

## 1) Setup local (Mac)

Depuis `training/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U datasets transformers peft trl bitsandbytes accelerate sentencepiece tqdm mistralai pydantic huggingface_hub "optimum[onnxruntime]"
```

Variables utiles:

```bash
export HF_TOKEN="hf_..."
export MISTRAL_API_KEY="..."
export TEACHER_MODEL="mistral-medium-latest"
export PROMPT_FILE="../src/lib/analysis_prompt.txt"
```

## 2) Pipeline locale en une commande

Option A: tout enchaîner (inclut build V1):

```bash
python run_local_teacher_pipeline.py --build-v1
```

Option B: partir d'un `train.jsonl` déjà construit:

```bash
python run_local_teacher_pipeline.py
```

Sorties:
- `candidates_10k.jsonl`
- `teacher_labeled_10k.jsonl`
- `train_v2.jsonl`
- `eval_v2.jsonl`

## 3) Détail des scripts locaux

- `build_v1_dataset.py`: construit `train.jsonl` / `eval.jsonl` avec `text = prompt + conversation + JSON`.
- `build_candidates_10k.py`: fait un échantillon équilibré (neg normal/hard + buckets positifs).
- `label_with_mistral_teacher.py`: labelise via API teacher en JSON structuré.
- `build_v2_from_teacher.py`: valide `teacher_label`, reconstruit `text` avec le prompt applicatif, split train/eval.

## 4) Passage sur Brev

Uploader minimum:
- `train_v2.jsonl`
- `eval_v2.jsonl`
- `train_q_lora.py`
- `evaluate.py`
- `export_onnx_and_push_hf.py`

Exemple:

```bash
scp train_v2.jsonl eval_v2.jsonl train_q_lora.py evaluate.py export_onnx_and_push_hf.py user@<brev-host>:/workspace/project/training/
```

## 5) Entraînement QLoRA sur Brev

Sur Brev:

```bash
cd /workspace/project/training
pip install -U datasets transformers peft trl bitsandbytes accelerate sentencepiece tqdm huggingface_hub "optimum[onnxruntime]"

export MODEL_NAME="mistralai/Ministral-3B-Instruct"
export TRAIN_FILE="train_v2.jsonl"
export EVAL_FILE="eval_v2.jsonl"
export OUTPUT_DIR="./lora-privacy"
export ADAPTER_DIR="./lora-privacy-adapter"
export LOGGING_DIR="./runs"
export REPORT_TO="tensorboard"

python train_q_lora.py
```

### Suivre l'entraînement

Le script sauvegarde:
- checkpoints dans `OUTPUT_DIR`
- adapter LoRA dans `ADAPTER_DIR`
- métriques dans `OUTPUT_DIR/training_metrics.json`
- logs TensorBoard dans `LOGGING_DIR`

Pour suivre en direct:

```bash
tensorboard --logdir ./runs --port 6006
```

## 6) Évaluation: `evaluate.py` est-il efficace?

Oui, maintenant il est utile pour une première évaluation quantitative:
- taux JSON valide
- accuracy verdict
- accuracy `low` vs `not-low`
- MAE du `overall_score`
- MAE par catégorie
- faux positifs sur négatifs (`overall_score >= 50`)
- comparaison optionnelle avec le modèle de base (`--compare-base`)

Exemple:

```bash
python evaluate.py \
  --model-name mistralai/Ministral-3B-Instruct \
  --adapter-path ./lora-privacy-adapter \
  --eval-file eval_v2.jsonl \
  --max-samples 300 \
  --compare-base \
  --report-file evaluation_report.json
```

Si tu veux aller plus loin: ajoute un petit set gold manuel (100-300 exemples) pour mesurer des écarts réellement utiles produit.

## 7) Export ONNX + upload Hugging Face

Script fourni: `export_onnx_and_push_hf.py`

Il fait:
1. merge LoRA + base
2. export ONNX avec `optimum-cli`
3. upload optionnel vers Hugging Face

Exemple complet:

```bash
export HF_TOKEN="hf_..."

python export_onnx_and_push_hf.py \
  --base-model mistralai/Ministral-3B-Instruct \
  --adapter-dir ./lora-privacy-adapter \
  --merged-dir ./merged-model \
  --onnx-dir ./onnx-export \
  --push-hf \
  --hf-repo-id <username>/ministral-privacy-onnx \
  --hf-private \
  --upload-adapter
```

Sorties locales:
- `./merged-model`
- `./onnx-export`

## 8) Commandes unitaires (si besoin)

```bash
# Build V1
python build_v1_dataset.py

# Candidates 10k
python build_candidates_10k.py --in train.jsonl --out candidates_10k.jsonl

# Teacher labeling
python label_with_mistral_teacher.py --in candidates_10k.jsonl --out teacher_labeled_10k.jsonl --resume

# Build V2
python build_v2_from_teacher.py --in teacher_labeled_10k.jsonl --train-out train_v2.jsonl --eval-out eval_v2.jsonl
```
