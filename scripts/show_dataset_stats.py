#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Show dataset composition stats")
    parser.add_argument("--dataset_dir", type=str, default="results/expert_dataset")
    return parser.parse_args()


def read_ids(path: Path):
    if not path.exists():
        return set()
    return {int(x.strip()) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()}


def main():
    args = parse_args()
    ds = Path(args.dataset_dir)
    meta = ds / "metadata.jsonl"
    train_ids = read_ids(ds / "splits" / "train_ids.txt")
    val_ids = read_ids(ds / "splits" / "val_ids.txt")

    if not meta.exists():
        raise SystemExit(f"Missing {meta}")

    action_all = Counter()
    action_train = Counter()
    action_val = Counter()
    env_all = Counter()

    total_rows = 0
    with meta.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total_rows += 1
            aid = int(row["action_id"])
            eid = int(row["episode_id"])
            env = str(row.get("env_id", "unknown"))

            action_all[aid] += 1
            env_all[env] += 1
            if eid in train_ids:
                action_train[aid] += 1
            if eid in val_ids:
                action_val[aid] += 1

    def fmt_counter(c: Counter):
        return {int(k): int(v) for k, v in sorted(c.items(), key=lambda x: x[0])}

    print(f"dataset_dir: {ds}")
    print(f"rows: {total_rows}")
    print(f"train episodes: {len(train_ids)} | val episodes: {len(val_ids)}")
    print(f"action counts all  : {fmt_counter(action_all)}")
    print(f"action counts train: {fmt_counter(action_train)}")
    print(f"action counts val  : {fmt_counter(action_val)}")
    print(f"env distribution   : {dict(env_all)}")

    nonzero_actions = sum(1 for _, v in action_all.items() if v > 0)
    if nonzero_actions < 3:
        print("[warning] Very low action diversity. Consider collect_envs with Random variants/sizes.")


if __name__ == "__main__":
    main()
