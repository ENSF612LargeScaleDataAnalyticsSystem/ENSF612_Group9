"""Paired bootstrap comparison of two runs on the same split and stage.

    python tools/compare.py results/knn/user_time/test results/popularity/user_time/test
    python tools/compare.py results/lsh/user_time/test results/knn/user_time/test --n-boot 2000

Both runs must have been evaluated on the same users (same split, stage and sample). For each metric
it prints mean(A) - mean(B), a 95% interval from resampling users (the same resample for A and B),
and the share of resamples where A is not better than B (a one-sided p-value).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import SEED, get_spark  # noqa: E402
from src.evaluate import METRICS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="run folder A (contains per_user/)")
    ap.add_argument("b", type=Path, help="run folder B")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--driver-memory", default="4g")
    args = ap.parse_args()

    spark = get_spark("compare", driver_memory=args.driver_memory)
    try:
        a = spark.read.parquet(str(args.a / "per_user")).select("user", *[spark_col(m, "a") for m in METRICS])
        b = spark.read.parquet(str(args.b / "per_user")).select("user", *[spark_col(m, "b") for m in METRICS])
        na, nb = a.count(), b.count()
        rows = a.join(b, "user").orderBy("user").collect()
    finally:
        spark.stop()
    if len(rows) != na or len(rows) != nb:
        raise SystemExit(f"runs evaluated different users ({na} vs {nb}, {len(rows)} shared): "
                         "compare runs on the same split, stage and sample")

    va = np.array([[r[f"{m}_a"] for m in METRICS] for r in rows])
    vb = np.array([[r[f"{m}_b"] for m in METRICS] for r in rows])
    diff = va - vb
    rng = np.random.default_rng(SEED)
    n = len(diff)
    boots = np.empty((args.n_boot, len(METRICS)))
    for i in range(args.n_boot):
        boots[i] = diff[rng.integers(0, n, n)].mean(axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    p = (boots <= 0).mean(axis=0)

    print(f"A = {args.a}\nB = {args.b}\n{n:,} users, {args.n_boot} paired resamples\n")
    print(f"{'metric':<10} {'A':>8} {'B':>8} {'A-B':>9}   95% interval          P(A<=B)")
    for j, m in enumerate(METRICS):
        print(f"{m:<10} {va[:, j].mean():8.4f} {vb[:, j].mean():8.4f} {diff[:, j].mean():+9.4f}"
              f"   [{lo[j]:+.4f}, {hi[j]:+.4f}]   {p[j]:.3f}")


def spark_col(m, side):
    from pyspark.sql import functions as F
    return F.col(m).alias(f"{m}_{side}")


if __name__ == "__main__":
    main()
