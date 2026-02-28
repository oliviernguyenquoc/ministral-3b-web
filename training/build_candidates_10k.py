import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


LOCATION_PATTERN = re.compile(
    r"\b("
    r"every day|every morning|every evening|commute|route|schedule|at \d{1,2}:\d{2}|"
    r"lives? at|home address|street|avenue|boulevard|zip code|postcode|"
    r"work at|office in|take the train|bus line|flight on|arrive at"
    r")\b",
    re.IGNORECASE,
)

HARD_NEG_PATTERN = re.compile(
    r"\b("
    r"salary|debt|loan|bank|bankruptcy|diagnosed|therapy|medication|"
    r"address|daily|every day|secret|cheat|fraud|lawsuit|"
    r"hospital|doctor|depressed|anxious"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class Reservoir:
    size: int

    def __post_init__(self):
        self.items: List[dict] = []
        self.seen = 0

    def add(self, item: dict) -> None:
        self.seen += 1
        if len(self.items) < self.size:
            self.items.append(item)
            return
        j = random.randint(0, self.seen - 1)
        if j < self.size:
            self.items[j] = item


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build balanced candidates_10k.jsonl from V1 train/eval.")
    p.add_argument("--in", dest="input_path", default="train.jsonl", help="Input V1 JSONL.")
    p.add_argument("--out", dest="output_path", default="candidates_10k.jsonl", help="Output candidates JSONL.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-total", type=int, default=10000)
    p.add_argument("--neg-normal", type=int, default=4000)
    p.add_argument("--neg-hard", type=int, default=2000)
    p.add_argument("--pos-health", type=int, default=800)
    p.add_argument("--pos-relationships", type=int, default=800)
    p.add_argument("--pos-finance", type=int, default=800)
    p.add_argument("--pos-confessions", type=int, default=800)
    p.add_argument("--pos-location", type=int, default=800)
    return p.parse_args()


def source_bucket(source: str) -> str | None:
    s = (source or "").lower()
    if "ultrachat_negative" in s:
        return "negative"
    if "mental_health" in s or "medical_dialog" in s:
        return "health"
    if "finance" in s or "kuvera" in s:
        return "finance"
    if "confession" in s or "tifu" in s:
        return "confessions"
    if "relationship" in s or "parenting" in s or "family" in s:
        return "relationships"
    return None


def conversation_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] skip malformed json line {line_no}")


def sample_unique(pool: List[dict], n: int, used_ids: set[str], rng: random.Random) -> List[dict]:
    if not pool or n <= 0:
        return []
    pool = pool[:]
    rng.shuffle(pool)
    out: List[dict] = []
    for x in pool:
        cid = conversation_id(x["conversation"])
        if cid in used_ids:
            continue
        out.append(x)
        used_ids.add(cid)
        if len(out) >= n:
            break
    return out


def materialize_example(ex: dict, bucket: str) -> dict | None:
    conv = ex.get("conversation")
    if not isinstance(conv, str) or not conv.strip():
        return None
    source = ex.get("source", "unknown")
    return {
        "conversation": conv.strip(),
        "source": source,
        "bucket": bucket,
        "meta": {"v1_source": source},
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    quotas: Dict[str, int] = {
        "negative_normal": args.neg_normal,
        "negative_hard": args.neg_hard,
        "positive_health": args.pos_health,
        "positive_relationships": args.pos_relationships,
        "positive_finance": args.pos_finance,
        "positive_confessions": args.pos_confessions,
        "positive_location": args.pos_location,
    }

    requested = sum(quotas.values())
    if requested != args.target_total:
        raise ValueError(f"Quota sum ({requested}) != --target-total ({args.target_total})")

    cap_mult = 4
    pools = {k: Reservoir(max(1, v * cap_mult)) for k, v in quotas.items()}
    fallback_pool = Reservoir(max(20000, args.target_total * 4))

    count_in = 0
    for row in iter_jsonl(input_path):
        count_in += 1
        src_bucket = source_bucket(str(row.get("source", "")))
        conversation = row.get("conversation", "")
        example_added = False

        if src_bucket == "negative":
            normal_ex = materialize_example(row, "negative_normal")
            if normal_ex:
                pools["negative_normal"].add(normal_ex)
                example_added = True
            if HARD_NEG_PATTERN.search(conversation or ""):
                hard_ex = materialize_example(row, "negative_hard")
                if hard_ex:
                    pools["negative_hard"].add(hard_ex)
                    example_added = True
        elif src_bucket == "health":
            ex = materialize_example(row, "positive_health")
            if ex:
                pools["positive_health"].add(ex)
                example_added = True
        elif src_bucket == "relationships":
            ex = materialize_example(row, "positive_relationships")
            if ex:
                pools["positive_relationships"].add(ex)
                example_added = True
        elif src_bucket == "finance":
            ex = materialize_example(row, "positive_finance")
            if ex:
                pools["positive_finance"].add(ex)
                example_added = True
        elif src_bucket == "confessions":
            ex = materialize_example(row, "positive_confessions")
            if ex:
                pools["positive_confessions"].add(ex)
                example_added = True

        if LOCATION_PATTERN.search(conversation or ""):
            ex = materialize_example(row, "positive_location")
            if ex:
                pools["positive_location"].add(ex)
                example_added = True

        if example_added:
            fallback = materialize_example(row, "fallback")
            if fallback:
                fallback_pool.add(fallback)

    print(f"[INFO] scanned rows: {count_in}")
    for k, reservoir in pools.items():
        print(f"[INFO] pool {k}: kept={len(reservoir.items)} seen={reservoir.seen}")
    print(f"[INFO] fallback pool: kept={len(fallback_pool.items)} seen={fallback_pool.seen}")

    selected: List[dict] = []
    used_ids: set[str] = set()
    for bucket, quota in quotas.items():
        picked = sample_unique(pools[bucket].items, quota, used_ids, rng)
        selected.extend(picked)
        if len(picked) < quota:
            missing = quota - len(picked)
            filler = sample_unique(fallback_pool.items, missing, used_ids, rng)
            for x in filler:
                x["bucket"] = f"{bucket}_fallback"
            selected.extend(filler)
            print(f"[WARN] bucket={bucket} missing={missing}, fallback_added={len(filler)}")

    if len(selected) < args.target_total:
        missing = args.target_total - len(selected)
        filler = sample_unique(fallback_pool.items, missing, used_ids, rng)
        for x in filler:
            x["bucket"] = "topup_fallback"
        selected.extend(filler)
        print(f"[WARN] global top-up added={len(filler)}")

    rng.shuffle(selected)
    selected = selected[: args.target_total]
    for i, row in enumerate(selected, start=1):
        row["id"] = f"cand-{i:06d}"

    with output_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {len(selected)} rows to {output_path}")


if __name__ == "__main__":
    main()
