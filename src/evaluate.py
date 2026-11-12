"""Step 4 - evaluation shared by all methods (proposal, Section 4.2).

Every method is scored the same way:
    * evaluable users = users with at least one liked item in the target part and at least one
      rating in the training part (ground_truth);
    * a user the method gives no list to still counts, with 0 for every metric;
    * relevance is binary: a recommended item is a hit if the user liked it in the target part;
    * Precision@k = hits / k, Recall@k = hits / (liked target items),
      NDCG@k = DCG / ideal DCG with DCG = sum over hits of 1 / log2(rank + 1);
    * coverage = distinct recommended items / items in the training part;
    * 95% confidence intervals by bootstrap over users.
"""
import math

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common import K, SEED

METRICS = ("precision", "recall", "ndcg")


def ground_truth(train: DataFrame, target: DataFrame) -> DataFrame:
    """(user, item) pairs the user liked in the target part, for users with training history."""
    users = train.select("user").distinct()
    return (target.where(F.col("like")).select("user", "item").distinct()
            .join(users, "user", "left_semi"))


def check_recs(recs: DataFrame, k: int, train: DataFrame = None) -> dict:
    """Format checks. Duplicates or bad ranks raise; recommending seen items only warns."""
    missing = {"user", "item", "rank"} - set(recs.columns)
    if missing:
        raise ValueError(f"recommendations are missing columns {missing}")
    row = recs.agg(
        F.count("*").alias("rows"),
        F.countDistinct("user", "item").alias("distinct_pairs"),
        F.countDistinct("user", "rank").alias("distinct_ranks"),
        F.sum(F.when((F.col("rank") < 1) | (F.col("rank") > k), 1).otherwise(0)).alias("bad_rank"),
        F.countDistinct("user").alias("users"),
    ).first().asDict()
    if row["rows"] != row["distinct_pairs"]:
        raise ValueError(f"{row['rows'] - row['distinct_pairs']} duplicate (user, item) rows in the recommendations")
    if row["rows"] != row["distinct_ranks"]:
        raise ValueError("some users have two items with the same rank")
    if row["bad_rank"]:
        raise ValueError(f"{row['bad_rank']} rows have a rank outside 1..{k}")
    out = {"rec_rows": row["rows"], "rec_users": row["users"]}
    if train is not None:
        seen = recs.join(train.select("user", "item"), ["user", "item"], "left_semi").count()
        out["recommended_items_already_rated"] = seen
        if seen:
            print(f"WARNING: {seen} recommended items were already rated by the user in training. "
                  "They count as misses; filter them out in the method.")
    return out


def per_user_metrics(recs: DataFrame, truth: DataFrame, k: int = K) -> DataFrame:
    """One row per evaluable user: user, n_relevant, hits, precision, recall, ndcg."""
    idcg_prefix, acc = [], 0.0
    for r in range(1, k + 1):
        acc += 1.0 / math.log2(r + 1)
        idcg_prefix.append(acc)
    n_rel = truth.groupBy("user").agg(F.count("*").alias("n_relevant"))
    hits = (recs.where(F.col("rank") <= k).join(truth, ["user", "item"])
            .groupBy("user").agg(F.count("*").alias("hits"),
                                 F.sum(1.0 / F.log2(F.col("rank") + 1.0)).alias("dcg")))
    idcg = F.element_at(F.array(*[F.lit(v) for v in idcg_prefix]), F.least(F.col("n_relevant"), F.lit(k)).cast("int"))
    return (n_rel.join(hits, "user", "left")
            .fillna({"hits": 0, "dcg": 0.0})
            .select("user", "n_relevant", "hits",
                    (F.col("hits") / F.lit(float(k))).alias("precision"),
                    (F.col("hits") / F.col("n_relevant")).alias("recall"),
                    (F.col("dcg") / idcg).alias("ndcg")))


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = SEED, alpha: float = 0.05):
    """Mean and percentile bootstrap interval for each column of `values` (users x metrics)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        means[b] = values[rng.integers(0, n, n)].mean(axis=0)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return values.mean(axis=0), lo, hi


def evaluate(recs: DataFrame, truth: DataFrame, train: DataFrame = None, k: int = K,
             n_boot: int = 1000, per_user_path: str = None) -> dict:
    """Score a top-k list. Pass `train` to check for seen items and to get the coverage catalogue."""
    result = {"k": k}
    result.update(check_recs(recs, k, train))
    pu = per_user_metrics(recs, truth, k)
    if per_user_path:
        pu.write.mode("overwrite").parquet(per_user_path)
        pu = pu.sparkSession.read.parquet(per_user_path)
    # Sorted by user so the bootstrap resamples the same rows on every machine.
    rows = pu.orderBy("user").select(*METRICS).collect()
    if not rows:
        raise ValueError("no evaluable users: is the target part empty?")
    values = np.array([[r[m] for m in METRICS] for r in rows], dtype=float)
    mean, lo, hi = bootstrap_ci(values, n_boot) if n_boot else (values.mean(axis=0), None, None)
    result["evaluated_users"] = len(values)
    result["evaluated_users_without_list"] = int(
        truth.select("user").distinct().join(recs.select("user").distinct(), "user", "left_anti").count())
    for j, m in enumerate(METRICS):
        result[f"{m}@{k}"] = round(float(mean[j]), 6)
        if lo is not None:
            result[f"{m}@{k}_ci95"] = [round(float(lo[j]), 6), round(float(hi[j]), 6)]
    if train is not None:
        n_catalog = train.select("item").distinct().count()
        n_rec_items = recs.where(F.col("rank") <= k).select("item").distinct().count()
        result["catalog_items"] = n_catalog
        result["coverage"] = round(n_rec_items / n_catalog, 6)
    return result
