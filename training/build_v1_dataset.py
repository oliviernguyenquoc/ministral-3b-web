import json, os, random, re
from pathlib import Path
from datasets import load_dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

random.seed(42)

# -----------------------------
# Config (env overridable)
# -----------------------------
PROMPT_FILE = Path(os.getenv("PROMPT_FILE", "../src/lib/analysis_prompt.txt"))
TRAIN_OUT = os.getenv("TRAIN_OUT", "train.jsonl")
EVAL_OUT = os.getenv("EVAL_OUT", "eval.jsonl")
EVAL_RATIO = float(os.getenv("EVAL_RATIO", "0.02"))
MIN_EVAL = int(os.getenv("MIN_EVAL", "1000"))

# Sizes
N_MENTAL_HEALTH = int(os.getenv("N_MENTAL_HEALTH", "20000"))
N_MEDICAL = int(os.getenv("N_MEDICAL", "20000"))  # will be skipped if dataset can't load
N_FINANCE_PF = int(os.getenv("N_FINANCE_PF", "20000"))
N_FINANCE_KU = int(os.getenv("N_FINANCE_KU", "20000"))
N_CONFESSIONS = int(os.getenv("N_CONFESSIONS", "20000"))
N_PROSOCIAL = int(os.getenv("N_PROSOCIAL", "15000"))

# Pushshift targets
PUSHSHIFT_STREAM_SCAN_MAX = int(os.getenv("PUSHSHIFT_STREAM_SCAN_MAX", "10000000"))
PUSHSHIFT_SHUFFLE_BUFFER = int(os.getenv("PUSHSHIFT_SHUFFLE_BUFFER", "200000"))
N_PUSH_REL = int(os.getenv("N_PUSH_REL", "30000"))
N_PUSH_FAM = int(os.getenv("N_PUSH_FAM", "30000"))
N_PUSH_TIFU = int(os.getenv("N_PUSH_TIFU", "15000"))

# Negatives (normal conversations)
N_NEG_ULTRACHAT = int(os.getenv("N_NEG_ULTRACHAT", "30000"))

# Text limits
MAX_CHARS = int(os.getenv("MAX_CHARS", "8000"))

CATS = [
  "health_vulnerability",
  "personal_relationships_conflicts",
  "financial_vulnerability",
  "risky_confessions_secrets",
  "location_routine_tracking",
]

# V1 uses weak provenance labels (coarse score only, not fine-grained scoring).
WEAK_POSITIVE_SCORE = int(os.getenv("WEAK_POSITIVE_SCORE", "40"))
WEAK_POSITIVE_RISK = os.getenv("WEAK_POSITIVE_RISK", "medium")

# Optional: HF token to avoid rate limits
HF_TOKEN = os.getenv("HF_TOKEN", None)

# -----------------------------
# Prompt loader
# -----------------------------
def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Set PROMPT_FILE or create ../src/lib/analysis_prompt.txt"
        )
    return path.read_text(encoding="utf-8").strip()

PROMPT = load_prompt(PROMPT_FILE)

# -----------------------------
# Utils
# -----------------------------
def pick_text_field(ds):
    # Works for non-streaming Dataset (indexable)
    first = ds[0] if len(ds) else {}
    for c in [
        "text","body","content","comment","message","prompt",
        "question","answer","response","context","selftext","title",
    ]:
        if c in ds.column_names and isinstance(first.get(c), str):
            return c
    for c in ds.column_names:
        if isinstance(first.get(c), str):
            return c
    return None

def norm(s: str) -> str:
    s = (s or "").replace("\r\n", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def clip(s: str, max_chars: int = MAX_CHARS) -> str:
    s = norm(s)
    return s if len(s) <= max_chars else s[:max_chars]

def to_chat(text: str) -> str:
    t = clip(text)
    return f"User: {t}\nAssistant:"

def to_chat_from_messages(messages):
    # messages: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}...]
    out = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = clip(m.get("content", ""))
        if not content:
            continue
        if role == "user":
            out.append(f"User: {content}")
        elif role == "assistant":
            out.append(f"Assistant: {content}")
        else:
            out.append(f"{role}: {content}")
    if not out or not out[-1].startswith("Assistant:"):
        out.append("Assistant:")
    return "\n".join(out)

def evidence_snippets(text: str, k=1, max_len=160):
    # No sensitivity heuristic: just short verbatim snippet(s)
    t = norm(text).replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return []
    if len(t) <= max_len:
        return [t]
    return [(t[:max_len-3] + "...")][:k]

def verdict_from_score(score: int) -> str:
    if score <= 24: return "low"
    if score <= 49: return "medium"
    if score <= 74: return "high"
    return "critical"

def make_label(target_cat: str, conversation: str):
    cats = {}
    for c in CATS:
        if c == target_cat:
            cats[c] = {
                "score": WEAK_POSITIVE_SCORE,
                "risk": WEAK_POSITIVE_RISK,
                "evidence": evidence_snippets(conversation, k=1),
            }
        else:
            cats[c] = {"score": 0, "risk": "none", "evidence": []}

    overall = WEAK_POSITIVE_SCORE
    verdict = verdict_from_score(overall)
    return {
        "verdict": verdict,
        "overall_score": overall,
        "summary": f"Weak provenance label indicates {target_cat}.",
        "categories": cats,
    }

def make_negative_label():
    cats = {c: {"score": 0, "risk": "none", "evidence": []} for c in CATS}
    return {
        "verdict": "low",
        "overall_score": 0,
        "summary": "No exploitable private disclosure detected.",
        "categories": cats,
    }

def wrap_example(conversation: str, label_json: dict, source: str):
    text = PROMPT + "\n\nConversation:\n" + conversation + "\n\nJSON:\n" + json.dumps(label_json, ensure_ascii=False)
    return {
        "source": source,
        "conversation": conversation,
        "label_json": json.dumps(label_json, ensure_ascii=False),
        "text": text,
    }

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_split(name: str, split: str = "train", streaming: bool = False):
    print(f"[LOAD] {name} ({split}) streaming={streaming}", flush=True)
    try:
        kwargs = {"split": split, "streaming": streaming}
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN
        ds = load_dataset(name, **kwargs)
    except Exception as e:
        print(f"[LOAD ERROR] {name}: {type(e).__name__}: {e}", flush=True)
        print(f"[SKIP] {name} unavailable, continuing without it", flush=True)
        return None

    if streaming:
        feats = getattr(ds, "features", {}) or {}
        print(f"[LOAD DONE] {name}: streaming=True cols={len(list(feats.keys()))}", flush=True)
    else:
        print(f"[LOAD DONE] {name}: rows={len(ds)} cols={len(ds.column_names)}", flush=True)
    return ds

def sample_rows(ds, n, desc):
    if ds is None:
        print(f"[SKIP] {desc}: dataset not loaded", flush=True)
        return

    tf = pick_text_field(ds)
    if tf is None:
        print(f"[WARN] {desc}: no text-like field found, skip", flush=True)
        return

    total = min(n, len(ds))
    idxs = random.sample(range(len(ds)), k=total)

    it = tqdm(idxs, total=total, desc=desc, unit="rows") if tqdm else idxs
    kept = 0
    for i in it:
        s = ds[i][tf]
        if isinstance(s, str) and s.strip():
            kept += 1
            yield s
    print(f"[DONE] {desc}: kept={kept}/{total} text_field={tf}", flush=True)

# -----------------------------
# Pushshift targeted sampling (streaming + shuffle + aliases)
# -----------------------------
def normalize_subreddit(s: str) -> str:
    s = (s or "").strip().lower()
    if s.startswith("r/"):
        s = s[2:]
    return s

SUB_ALIASES = {
    # relationships is often in relationship_advice
    "relationships": {"relationships", "relationship_advice", "dating_advice", "dating"},
    "parenting_family": {"parenting", "family", "dadit", "mommit"},
    "tifu_todayifuckedup": {"tifu", "todayifuckedup"},
}

def sample_pushshift_targeted(
    ds,
    targets: dict,
    subreddit_aliases: dict,
    text_field="body",
    sub_col="subreddit",
    buffer_shuffle=200_000,
    max_scan=10_000_000,
):
    """
    Stream + shuffle + collect until each bucket reaches its target.
    """
    if ds is None:
        print("[SKIP] PUSHSHIFT: dataset not loaded", flush=True)
        return {k: [] for k in targets}

    # Shuffle streaming to avoid shard/time bias
    try:
        ds = ds.shuffle(seed=42, buffer_size=buffer_shuffle)
        print(f"[PUSHSHIFT] shuffle enabled buffer_size={buffer_shuffle}", flush=True)
    except Exception as e:
        print(f"[PUSHSHIFT WARN] shuffle not available: {type(e).__name__}: {e}", flush=True)

    buckets = {k: [] for k in targets}
    scanned = 0

    def done():
        return all(len(buckets[k]) >= targets[k] for k in targets)

    iterator = tqdm(ds, total=max_scan, desc="PUSHSHIFT/scan", unit="rows") if tqdm else ds

    for ex in iterator:
        scanned += 1
        if scanned > max_scan or done():
            break

        if not isinstance(ex, dict):
            continue

        txt = ex.get(text_field)
        if not isinstance(txt, str) or not txt.strip():
            continue

        sub = normalize_subreddit(ex.get(sub_col)) if sub_col else ""

        for bucket_name, subs in subreddit_aliases.items():
            if len(buckets[bucket_name]) >= targets[bucket_name]:
                continue
            if (not sub_col) or (sub in subs):
                buckets[bucket_name].append(txt)

    for k in buckets:
        random.shuffle(buckets[k])
        print(f"[DONE] PUSHSHIFT/{k}: kept={len(buckets[k])}/{targets[k]} scanned={scanned}", flush=True)

    return buckets

# -----------------------------
# Main
# -----------------------------
def main():
    print("[START] build_v1_dataset", flush=True)
    rows = []

    # HEALTH
    mh = load_split("solomonk/reddit_mental_health_posts", split="train", streaming=False)

    # medical_dialog currently fails on latest datasets versions (script-based). keep graceful skip.
    med = load_split("UCSD26/medical_dialog", split="train", streaming=False)

    for s in sample_rows(mh, N_MENTAL_HEALTH, "HEALTH/reddit_mental_health_posts"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("health_vulnerability", conv), "reddit_mental_health_posts"))

    if med is not None:
        for s in sample_rows(med, N_MEDICAL, "HEALTH/medical_dialog"):
            conv = to_chat(s)
            rows.append(wrap_example(conv, make_label("health_vulnerability", conv), "medical_dialog"))
    else:
        print("[INFO] medical_dialog skipped (not loadable in this environment).", flush=True)

    print(f"[STATUS] after HEALTH: rows={len(rows)}", flush=True)

    # FINANCE
    pf = load_split("Akhil-Theerthala/Personal-Finance-Queries", split="train", streaming=False)
    ku = load_split("Akhil-Theerthala/Kuvera-PersonalFinance-V2.1", split="train", streaming=False)

    for s in sample_rows(pf, N_FINANCE_PF, "FINANCE/Personal-Finance-Queries"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("financial_vulnerability", conv), "Personal-Finance-Queries"))

    for s in sample_rows(ku, N_FINANCE_KU, "FINANCE/Kuvera-PersonalFinance-V2.1"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("financial_vulnerability", conv), "Kuvera-PersonalFinance-V2.1"))

    print(f"[STATUS] after FINANCE: rows={len(rows)}", flush=True)

    # SECRETS
    conf = load_split("SocialGrep/one-million-reddit-confessions", split="train", streaming=False)
    pro = load_split("shahules786/prosocial-confessions", split="train", streaming=False)

    for s in sample_rows(conf, N_CONFESSIONS, "SECRETS/one-million-reddit-confessions"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("risky_confessions_secrets", conv), "one-million-reddit-confessions"))

    for s in sample_rows(pro, N_PROSOCIAL, "SECRETS/prosocial-confessions"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("risky_confessions_secrets", conv), "prosocial-confessions"))

    print(f"[STATUS] after SECRETS: rows={len(rows)}", flush=True)

    # PUSHSHIFT (streaming targeted)
    ps = load_split("fddemarco/pushshift-reddit-comments", split="train", streaming=True)
    ps_features = getattr(ps, "features", {}) or {}
    sub_col = "subreddit" if "subreddit" in ps_features else None
    text_field = "body" if "body" in ps_features else None

    if ps is None or text_field is None:
        print("[WARN] PUSHSHIFT missing or no 'body' field, skipping PUSHSHIFT.", flush=True)
        push_samples = {"relationships": [], "parenting_family": [], "tifu_todayifuckedup": []}
    else:
        print(f"[INFO] PUSHSHIFT streaming: text_field={text_field} sub_col={sub_col}", flush=True)
        push_samples = sample_pushshift_targeted(
            ps,
            targets={
                "relationships": N_PUSH_REL,
                "parenting_family": N_PUSH_FAM,
                "tifu_todayifuckedup": N_PUSH_TIFU,
            },
            subreddit_aliases=SUB_ALIASES,
            text_field=text_field,
            sub_col=sub_col,
            buffer_shuffle=PUSHSHIFT_SHUFFLE_BUFFER,
            max_scan=PUSHSHIFT_STREAM_SCAN_MAX,
        )

    for s in push_samples["relationships"]:
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("personal_relationships_conflicts", conv), "pushshift_relationships"))

    for s in push_samples["parenting_family"]:
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("personal_relationships_conflicts", conv), "pushshift_parenting_family"))

    for s in push_samples["tifu_todayifuckedup"]:
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("risky_confessions_secrets", conv), "pushshift_tifu"))

    print(f"[STATUS] after PUSHSHIFT: rows={len(rows)}", flush=True)

    # NORMAL / NEGATIVE (UltraChat)
    uc = load_split("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=False)
    if uc is not None:
        total = min(N_NEG_ULTRACHAT, len(uc))
        idxs = random.sample(range(len(uc)), k=total)
        it = tqdm(idxs, total=total, desc="NEG/ultrachat_200k", unit="rows") if tqdm else idxs
        kept = 0
        neg_label = make_negative_label()
        for i in it:
            msgs = uc[i].get("messages")
            if isinstance(msgs, list) and msgs:
                conv = to_chat_from_messages(msgs)
                rows.append(wrap_example(conv, neg_label, "ultrachat_negative"))
                kept += 1
        print(f"[DONE] NEG/ultrachat_200k: kept={kept}/{total}", flush=True)
    else:
        print("[WARN] UltraChat not loaded; you currently have almost no negatives.", flush=True)

    print(f"[STATUS] after NEGATIVES: rows={len(rows)}", flush=True)

    # Shuffle + split
    print("[STEP] shuffle + split train/eval", flush=True)
    random.shuffle(rows)
    n_eval = max(MIN_EVAL, int(EVAL_RATIO * len(rows)))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    print(f"[STEP] write {TRAIN_OUT}", flush=True)
    write_jsonl(TRAIN_OUT, train_rows)
    print(f"[STEP] write {EVAL_OUT}", flush=True)
    write_jsonl(EVAL_OUT, eval_rows)
    print(f"[DONE] train={len(train_rows)} eval={len(eval_rows)} total={len(rows)}", flush=True)

if __name__ == "__main__":
    main()