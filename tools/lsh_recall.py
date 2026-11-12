"""How close MinHash-LSH neighbours are to the exact ones (RQ: accuracy vs cost of approximation).

    python tools/lsh_recall.py --split user_time_10pct --n-hashes 128 --bands 64
    python tools/lsh_recall.py --split user_time --n-hashes 128 --bands 64 --users 2000

For a random sample of users (common.SEED) it computes the exact top-N neighbours (knn.py) and the
LSH top-N neighbours (lsh.py, same similarity), then reports
  * neighbour recall@N: |LSH top-N ∩ exact top-N| / |exact top-N|, averaged over users;
  * similarity ratio:   sum of LSH neighbours' similarity / sum of exact neighbours' similarity;
  * candidates per user and the time of the candidate step.
Uses the training part of the val stage, like tuning does.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import SEED, get_spark, load_stage  # noqa: E402
from src.methods import knn, lsh  # noqa: E402
from src.methods._shared import build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="user_time_10pct")
    ap.add_argument("--n-hashes", type=int, default=128)
    ap.add_argument("--bands", type=int, default=64)
    ap.add_argument("--max-bucket", type=int, default=2000)
    ap.add_argument("--n-neighbours", type=int, default=50)
    ap.add_argument("--similarity", default="jaccard")
    ap.add_argument("--users", type=int, default=1000, help="users sampled for the comparison")
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()

    spark = get_spark("lsh-recall", driver_memory=args.driver_memory)
    try:
        train, _ = load_stage(spark, args.split, "val")
        d = build_index(train)
        bc = spark.sparkContext.broadcast(d)
        t0 = time.time()
        per_user = lsh.candidates(spark, d, bc, args.n_hashes, args.bands, args.max_bucket, SEED)
        rng = np.random.default_rng(SEED)
        sample = set(rng.choice(d["users"], size=min(args.users, len(d["users"])), replace=False).tolist())
        rows = per_user.where(per_user.user.isin(list(sample))).collect()
        t_cand = time.time() - t0
    finally:
        spark.stop()

    n = args.n_neighbours
    recalls, ratios, n_cands = [], [], []
    for r in rows:
        u = r["user"]
        ex_n, ex_s = knn.neighbours_exact(u, d, n, args.similarity)
        if len(ex_n) == 0:
            continue
        c = np.asarray(r["cands"] or [], np.int64)
        ap_n, ap_s = lsh.candidate_scores(u, c, d, n, args.similarity)
        recalls.append(len(set(ex_n.tolist()) & set(ap_n.tolist())) / len(ex_n))
        ratios.append(ap_s.sum() / ex_s.sum())
        n_cands.append(max(len(c) - 1, 0))

    rows_per_band = args.n_hashes // args.bands
    print(f"{args.split}: n_hashes={args.n_hashes} bands={args.bands} (r={rows_per_band}, threshold ~ "
          f"{(1 / args.bands) ** (1 / rows_per_band):.3f}) max_bucket={args.max_bucket}")
    print(f"  users compared            {len(recalls)}")
    print(f"  neighbour recall@{n:<8} {np.mean(recalls):.4f}")
    print(f"  similarity ratio          {np.mean(ratios):.4f}")
    print(f"  candidates per user       {np.mean(n_cands):.1f} (median {np.median(n_cands):.0f})")
    print(f"  candidate step            {t_cand:.1f}s (all users)")


if __name__ == "__main__":
    main()
