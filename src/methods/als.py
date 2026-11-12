"""ALS matrix factorisation (Spark MLlib), implicit feedback by default.

signal='like'   (default) every liked (user, movie) pair is a positive with value 1; implicit ALS
                treats all other pairs as weak negatives (Hu, Koren and Volinsky 2008).
signal='rating' every rating is used, its value as the confidence (implicit) or target (explicit).

Scoring is exact: item factors are broadcast and each batch of users computes U V^T with numpy,
removes the movies the user rated and keeps the k best. (recommendForAllUsers cannot leave out
seen movies, so it would need k + 9,254 candidates per user.) numUserBlocks/numItemBlocks are
fixed so the factors do not change with the number of cores.

Params: rank (64), reg (0.05), alpha (10.0, implicit confidence), max_iter (10),
        implicit (True), signal ('like' or 'rating'), blocks (10), seed (common.SEED).
"""
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common import SEED
from src.methods._shared import build_index, ranked_rows, row, run_batches, top_k, user_batches


def _factors(df: DataFrame, n: int, rank: int):
    """Factor matrix (n, rank) and a mask of the ids that have a factor."""
    m = np.zeros((n, rank), np.float32)
    has = np.zeros(n, np.bool_)
    for r in df.collect():
        if r["id"] < n:
            m[r["id"]] = r["features"]
            has[r["id"]] = True
    return m, has


def recommend(spark: SparkSession, train: DataFrame, k: int = 10, rank: int = 64, reg: float = 0.05,
              alpha: float = 10.0, max_iter: int = 10, implicit: bool = True, signal: str = "like",
              blocks: int = 10, seed: int = SEED) -> DataFrame:
    if signal == "like":
        data = train.where(F.col("like")).select("user", "item", F.lit(1.0).alias("r"))
    elif signal == "rating":
        data = train.select("user", "item", F.col("rating").alias("r"))
    else:
        raise ValueError("signal must be 'like' or 'rating'")

    als = ALS(rank=int(rank), regParam=float(reg), alpha=float(alpha), maxIter=int(max_iter),
              implicitPrefs=bool(implicit), userCol="user", itemCol="item", ratingCol="r",
              numUserBlocks=int(blocks), numItemBlocks=int(blocks), seed=int(seed),
              coldStartStrategy="drop", nonnegative=False)
    model = als.fit(data)

    d = build_index(train)
    # users without a factor (no like when signal='like') get the popular items
    d["U"], d["user_has"] = _factors(model.userFactors, d["n_users"], int(rank))
    d["V"], d["has_factor"] = _factors(model.itemFactors, d["n_items"], int(rank))

    def score_batch(batch, d):
        scores = d["U"][batch] @ d["V"].T
        out = []
        for j, u in enumerate(batch):
            seen = row(d["rated_ptr"], d["rated_idx"], u)
            valid = d["has_factor"] if d["user_has"][u] else np.zeros_like(d["has_factor"])
            out.extend(ranked_rows(u, top_k(scores[j], seen, k, d["pop_order"], valid=valid)))
        return out

    return run_batches(spark, user_batches(d["users"]), score_batch, d)
