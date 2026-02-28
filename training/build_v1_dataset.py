import json, os, random, re
from pathlib import Path
from datasets import load_dataset

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

random.seed(42)

PROMPT_FILE = Path(os.getenv("PROMPT_FILE", "../src/lib/analysis_prompt.txt"))

CATS = [
  "health_vulnerability",
  "personal_relationships_conflicts",
  "financial_vulnerability",
  "risky_confessions_secrets",
  "location_routine_tracking",
]

# V1 uses weak provenance labels (coarse score only, not fine-grained scoring).
WEAK_POSITIVE_SCORE = 40
WEAK_POSITIVE_RISK = "medium"

def load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. Set PROMPT_FILE or create ../src/lib/analysis_prompt.txt"
        )
    return path.read_text(encoding="utf-8").strip()

PROMPT = load_prompt(PROMPT_FILE)

def pick_text_field(ds):
    first = ds[0] if len(ds) else {}
    for c in [
        "text",
        "body",
        "content",
        "comment",
        "message",
        "prompt",
        "question",
        "answer",
        "response",
        "context",
        "selftext",
        "title",
    ]:
        if c in ds.column_names and isinstance(first.get(c), str):
            return c
    for c in ds.column_names:
        if isinstance(first.get(c), str):
            return c
    return None

def norm(s: str) -> str:
    s = (s or "").replace("\r\n","\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def to_chat(text: str) -> str:
    t = norm(text)
    return f"User: {t}\nAssistant:"

def evidence_snippets(text: str, k=1, max_len=160):
    # Pas d’heuristique “sensibilité”. Juste des extraits verbatim.
    t = norm(text).replace("\n", " ")
    t = re.sub(r"\s+"," ", t).strip()
    if not t: return []
    if len(t) <= max_len:
        return [t]
    # Simple weak evidence: keep one short verbatim snippet.
    out = []
    out.append(t[:max_len-3] + "...")
    if len(t) > 2*max_len:
        mid = len(t)//2
        start = max(0, mid - (max_len//2))
        chunk = t[start:start+max_len]
        out.append(chunk[:max_len-3] + "..." if len(chunk) == max_len else chunk)
    return out[:k]

def verdict_from_score(score: int) -> str:
    if score <= 24:
        return "low"
    if score <= 49:
        return "medium"
    if score <= 74:
        return "high"
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
        "summary": f"Weak label indicates {target_cat}.",
        "categories": cats,
    }

def wrap_example(conversation: str, label_json: dict):
    text = PROMPT + "\n\nConversation:\n" + conversation + "\n\nJSON:\n" + json.dumps(label_json, ensure_ascii=False)
    return {"conversation": conversation, "label_json": json.dumps(label_json, ensure_ascii=False), "text": text}

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_split(name: str, split: str = "train", streaming: bool = False):
    print(f"[LOAD] {name} ({split})", flush=True)
    try:
        ds = load_dataset(name, split=split, streaming=streaming)
    except Exception as e:
        print(f"[LOAD ERROR] {name}: {type(e).__name__}: {e}", flush=True)
        print(f"[SKIP] {name} unavailable, continuing without it", flush=True)
        return None
    if streaming:
        cols = list((getattr(ds, "features", {}) or {}).keys())
        print(f"[LOAD DONE] {name}: streaming=True cols={len(cols)}", flush=True)
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
    valid = 0

    if tqdm is not None:
        iterator = tqdm(idxs, total=total, desc=desc, unit="rows")
    else:
        iterator = idxs
        print(f"[START] {desc}: total_candidates={total}, text_field={tf}", flush=True)
        log_every = max(1, total // 20) if total else 1

    for j, i in enumerate(iterator, start=1):
        s = ds[i][tf]
        if isinstance(s, str) and s.strip():
            valid += 1
            yield s
        if tqdm is None and (j % log_every == 0 or j == total):
            pct = (j / total) * 100 if total else 100.0
            print(f"[PROGRESS] {desc}: {j}/{total} ({pct:.1f}%)", flush=True)

    print(f"[DONE] {desc}: kept={valid}/{total}", flush=True)

def sample_pushshift_rows(ds, targets, text_field="body", sub_col="subreddit", max_scan=1_500_000):
    """
    Stream and reservoir-sample multiple subreddit buckets in one pass, to avoid full dataset download.
    """
    if ds is None:
        print("[SKIP] PUSHSHIFT: dataset not loaded", flush=True)
        return {k: [] for k in targets}

    buckets = {k: [] for k in targets}
    matched = {k: 0 for k in targets}
    scanned = 0

    if tqdm is not None:
        iterator = tqdm(ds, total=max_scan, desc="PUSHSHIFT/stream_scan", unit="rows")
    else:
        iterator = ds
        print(
            f"[START] PUSHSHIFT/stream_scan: max_scan={max_scan}, text_field={text_field}, sub_col={sub_col}",
            flush=True,
        )
        log_every = max(1, max_scan // 20)

    def reservoir_add(bucket_name, value):
        matched[bucket_name] += 1
        target = targets[bucket_name]
        slot = buckets[bucket_name]
        if len(slot) < target:
            slot.append(value)
            return
        j = random.randint(1, matched[bucket_name])
        if j <= target:
            slot[j - 1] = value

    for ex in iterator:
        scanned += 1
        s = ex.get(text_field) if isinstance(ex, dict) else None
        if isinstance(s, str) and s.strip():
            sub = (ex.get(sub_col) or "").lower() if sub_col else ""
            if (not sub_col) or sub == "relationships":
                reservoir_add("relationships", s)
            if (not sub_col) or sub in {"parenting", "family"}:
                reservoir_add("parenting_family", s)
            if (not sub_col) or sub in {"tifu", "todayifuckedup"}:
                reservoir_add("tifu_todayifuckedup", s)

        if scanned >= max_scan:
            break
        if tqdm is None and (scanned % log_every == 0):
            print(f"[PROGRESS] PUSHSHIFT/stream_scan: scanned={scanned}/{max_scan}", flush=True)

    if tqdm is not None and hasattr(iterator, "close"):
        iterator.close()

    for key in buckets:
        random.shuffle(buckets[key])
        print(
            f"[DONE] PUSHSHIFT/{key}: matched={matched[key]} kept={len(buckets[key])}/{targets[key]}",
            flush=True,
        )
    print(f"[DONE] PUSHSHIFT/stream_scan: scanned={scanned}", flush=True)
    return buckets

def main():
    print("[START] build_v1_dataset", flush=True)
    rows = []

    # HEALTH
    mh = load_split("solomonk/reddit_mental_health_posts", split="train")
    med = load_split("UCSD26/medical_dialog", split="train")
    for s in sample_rows(mh, 20000, "HEALTH/reddit_mental_health_posts"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("health_vulnerability", conv)))
    for s in sample_rows(med, 20000, "HEALTH/medical_dialog"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("health_vulnerability", conv)))
    print(f"[STATUS] after HEALTH: rows={len(rows)}", flush=True)

    # FINANCE
    pf = load_split("Akhil-Theerthala/Personal-Finance-Queries", split="train")
    ku = load_split("Akhil-Theerthala/Kuvera-PersonalFinance-V2.1", split="train")
    for s in sample_rows(pf, 20000, "FINANCE/Personal-Finance-Queries"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("financial_vulnerability", conv)))
    for s in sample_rows(ku, 20000, "FINANCE/Kuvera-PersonalFinance-V2.1"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("financial_vulnerability", conv)))
    print(f"[STATUS] after FINANCE: rows={len(rows)}", flush=True)

    # SECRETS
    conf = load_split("SocialGrep/one-million-reddit-confessions", split="train")
    pro = load_split("shahules786/prosocial-confessions", split="train")
    for s in sample_rows(conf, 20000, "SECRETS/one-million-reddit-confessions"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("risky_confessions_secrets", conv)))
    for s in sample_rows(pro, 20000, "SECRETS/prosocial-confessions"):
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("risky_confessions_secrets", conv)))
    print(f"[STATUS] after SECRETS: rows={len(rows)}", flush=True)

    # PUSHSHIFT: streaming sample to avoid downloading full multi-TB corpus
    ps = load_split("fddemarco/pushshift-reddit-comments", split="train", streaming=True)
    ps_features = getattr(ps, "features", {}) or {}
    sub_col = "subreddit" if "subreddit" in ps_features else None
    text_field = "body" if "body" in ps_features else None

    if sub_col:
        print("[INFO] PUSHSHIFT subreddit column detected; streaming per subreddit buckets", flush=True)
    else:
        print("[WARN] PUSHSHIFT has no subreddit column; fallback to unfiltered stream", flush=True)
    if text_field is None:
        print("[WARN] PUSHSHIFT has no body field; skip PUSHSHIFT block", flush=True)
        push_samples = {
            "relationships": [],
            "parenting_family": [],
            "tifu_todayifuckedup": [],
        }
    else:
        push_samples = sample_pushshift_rows(
            ps,
            targets={
                "relationships": 30000,
                "parenting_family": 30000,
                "tifu_todayifuckedup": 15000,
            },
            text_field=text_field,
            sub_col=sub_col,
            max_scan=1_500_000,
        )

    for s in push_samples["relationships"]:
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("personal_relationships_conflicts", conv)))
    for s in push_samples["parenting_family"]:
        conv = to_chat(s)
        rows.append(wrap_example(conv, make_label("personal_relationships_conflicts", conv)))
    for s in push_samples["tifu_todayifuckedup"]:
        conv = to_chat(s)
        # tifu : tu as mis dans Secrets, je garde ton choix V1
        rows.append(wrap_example(conv, make_label("risky_confessions_secrets", conv)))
    print(f"[STATUS] after PUSHSHIFT: rows={len(rows)}", flush=True)

    print("[STEP] shuffle + split train/eval", flush=True)
    random.shuffle(rows)
    n_eval = max(1000, int(0.02 * len(rows)))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    print("[STEP] write train.jsonl", flush=True)
    write_jsonl("train.jsonl", train_rows)
    print("[STEP] write eval.jsonl", flush=True)
    write_jsonl("eval.jsonl", eval_rows)
    print(f"train={len(train_rows)} eval={len(eval_rows)}")

if __name__ == "__main__":
    main()
