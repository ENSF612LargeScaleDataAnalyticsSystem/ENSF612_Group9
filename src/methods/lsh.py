"""MinHash-LSH user-kNN: approximate neighbours, then the same scoring as exact kNN (knn.py).

  1. MinHash signature of each user's liked set with n_hashes hash functions
     h(x) = (a*x + b) mod p, p = 2^31 - 1.
  2. Banding: the signature is cut into `bands` bands of r = n_hashes / bands rows. Two users are
     candidates if they agree on all r values of at least one band, so a pair with Jaccard s is
     found with probability 1 - (1 - s^r)^b.
  3. Buckets with more than max_bucket users are skipped (they come from very popular movies and
     would bring back the quadratic cost LSH is meant to avoid); the number skipped is printed.
  4. The candidates' exact Jaccard (or cosine) is computed, the n_neighbours best are kept, and
     items are scored exactly as in knn.py. Only the neighbour search differs between the two.

Params: n_hashes (128), bands (32), n_neighbours (50), max_bucket (2000),
        similarity ('jaccard' or 'cosine'), seed (common.SEED).
"""
import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common import SEED
from src.methods._shared import (REC_SCHEMA, best_n, build_index, gather, neighbour_scores,
                                 ranked_rows, row, top_k, user_batches)
from src.methods._shared import similarity as sim_fn

PRIME = 2_147_483_647


def hash_params(n_hashes: int, bands: int, seed: int):
    if n_hashes % bands:
        raise ValueError(f"n_hashes ({n_hashes}) must be a multiple of bands ({bands})")
    rng = np.random.default_rng(seed)
    a = rng.integers(1, PRIME, n_hashes, dtype=np.int64)
    b = rng.integers(0, PRIME, n_hashes, dtype=np.int64)
    mult = rng.integers(1, 2 ** 62, n_hashes // bands, dtype=np.int64) | 1   # combines a band into one key
    return a, b, mult


def signature(items: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = items.astype(np.int64)
    return ((a[:, None] * x[None, :] + b[:, None]) % PRIME).min(axis=1)


def band_keys(sig: np.ndarray, bands: int, mult: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):                     # int64 wrap-around is fine for a hash
        return (sig.reshape(bands, -1) * mult[None, :]).sum(axis=1)


def candidate_scores(u: int, cands: np.ndarray, d: dict, n: int, kind: str):
    """Exact similarity of u to its LSH candidates, best n kept."""
    liked = row(d["liked_ptr"], d["liked_idx"], u)
    cands = cands[cands != u]
    if len(liked) == 0 or len(cands) == 0:
        return np.empty(0, np.int32), np.empty(0)
    mark = np.zeros(d["n_items"], np.bool_)
    mark[liked] = True
    items, lens = gather(d["liked_ptr"], d["liked_idx"], cands)
    inter = np.bincount(np.repeat(np.arange(len(cands)), lens), weights=mark[items], minlength=len(cands))
    keep = inter > 0
    cands, inter, lens = cands[keep], inter[keep], lens[keep]
    return best_n(cands.astype(np.int32), sim_fn(inter, len(liked), lens, kind), n)


def candidates(spark: SparkSession, d: dict, bc, n_hashes: int, bands: int, max_bucket: int,
               seed: int) -> DataFrame:
    """(user, cands): every user in train and the users sharing at least one band bucket with them
    (the user itself included, null if none). Cached; prints bucket and candidate statistics."""
    sc = spark.sparkContext
    a, b, mult = hash_params(int(n_hashes), int(bands), int(seed))
    n_bands = int(bands)
    batches = user_batches(d["users"])
    n_slices = max(1, min(len(batches), sc.defaultParallelism * 4))

    def keys_of(it):
        dd = bc.value
        for batch in it:
            for u in batch:
                liked = row(dd["liked_ptr"], dd["liked_idx"], u)
                if len(liked):
                    for j, key in enumerate(band_keys(signature(liked, a, b), n_bands, mult)):
                        yield (j, int(key), int(u))

    keys = spark.createDataFrame(sc.parallelize(batches, n_slices).mapPartitions(keys_of),
                                 "band INT, key LONG, user INT")
    buckets = (keys.groupBy("band", "key").agg(F.collect_list("user").alias("m"))
               .withColumn("size", F.size("m")).where(F.col("size") >= 2)).cache()
    big = buckets.where(F.col("size") > max_bucket).agg(F.count("*"), F.sum("size")).first()
    if big[0]:
        print(f"lsh: skipped {big[0]} buckets with more than {max_bucket} users ({big[1]} memberships)")

    cands = (buckets.where(F.col("size") <= max_bucket)
             .select(F.explode("m").alias("user"), "m")
             .groupBy("user").agg(F.array_distinct(F.flatten(F.collect_list("m"))).alias("cands")))
    users = spark.createDataFrame([(int(u),) for u in d["users"]], "user INT")
    per_user = users.join(cands, "user", "left").cache()
    st = per_user.agg(F.avg(F.coalesce(F.size("cands") - 1, F.lit(0))),
                      F.sum(F.when(F.col("cands").isNull(), 1).otherwise(0))).first()
    print(f"lsh: {st[0]:.1f} candidates per user on average; {st[1]} users without candidates")
    buckets.unpersist(blocking=False)
    return per_user


def recommend(spark: SparkSession, train: DataFrame, k: int = 10, n_hashes: int = 128, bands: int = 32,
              n_neighbours: int = 50, max_bucket: int = 2000, similarity: str = "jaccard",
              seed: int = SEED) -> DataFrame:
    d = build_index(train)
    bc = spark.sparkContext.broadcast(d)
    n, kind = int(n_neighbours), similarity
    per_user = candidates(spark, d, bc, n_hashes, bands, max_bucket, seed)

    def score(it):
        dd = bc.value
        for r in it:
            u = r["user"]
            c = np.asarray(r["cands"] or [], np.int64)
            nbrs, sims = candidate_scores(u, c, dd, n, kind)
            seen = row(dd["rated_ptr"], dd["rated_idx"], u)
            yield from ranked_rows(u, top_k(neighbour_scores(nbrs, sims, dd), seen, k, dd["pop_order"]))

    return spark.createDataFrame(per_user.rdd.mapPartitions(score), REC_SCHEMA)
