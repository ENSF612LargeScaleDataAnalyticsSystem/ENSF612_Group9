"""Exact user-kNN on liked-item sets: the reference that MinHash-LSH approximates.

For a user u with liked set A_u:
  1. similarity to every other user v with |A_u ∩ A_v| > 0, computed exactly. Intersections come
     from the item -> likers index: concatenating the likers of every item in A_u and counting each
     user gives |A_u ∩ A_v| for all v in one bincount.
  2. keep the n_neighbours most similar users (ties: lower id);
  3. score(i) = sum of sim(u, v) over those neighbours v who liked i; recommend the k best items
     u has not rated. Users with no like, or too few scored items, are filled with popular items.

Cost per user is the sum of the like counts of the items u liked, so the total is sum_i n_i^2.
That is why an exact all-pairs join does not scale, and why LSH is worth comparing against.

Params: n_neighbours (default 50), similarity ('jaccard' default, or 'cosine'),
        min_overlap (default 1: neighbours must share at least this many liked items).
"""
import numpy as np
from pyspark.sql import DataFrame, SparkSession

from src.methods._shared import (best_n, build_index, gather, neighbour_scores, ranked_rows, row,
                                 run_batches, top_k, user_batches)
from src.methods._shared import similarity as sim_fn


def neighbours_exact(u: int, d: dict, n: int, kind: str, min_overlap: int = 1):
    liked = row(d["liked_ptr"], d["liked_idx"], u)
    if len(liked) == 0:
        return np.empty(0, np.int32), np.empty(0)
    cand, _ = gather(d["likers_ptr"], d["likers_idx"], liked)
    counts = np.bincount(cand, minlength=d["n_users"])
    counts[u] = 0
    nbrs = np.flatnonzero(counts >= max(1, min_overlap))
    deg = d["liked_ptr"][nbrs + 1] - d["liked_ptr"][nbrs]
    sims = sim_fn(counts[nbrs], len(liked), deg, kind)
    return best_n(nbrs.astype(np.int32), sims, n)


def recommend(spark: SparkSession, train: DataFrame, k: int = 10, n_neighbours: int = 50,
              similarity: str = "jaccard", min_overlap: int = 1) -> DataFrame:
    d = build_index(train)
    kind, n, mo = similarity, int(n_neighbours), int(min_overlap)

    def score_batch(batch, d):
        out = []
        for u in batch:
            nbrs, sims = neighbours_exact(int(u), d, n, kind, mo)
            scores = neighbour_scores(nbrs, sims, d)
            seen = row(d["rated_ptr"], d["rated_idx"], u)
            out.extend(ranked_rows(u, top_k(scores, seen, k, d["pop_order"])))
        return out

    return run_batches(spark, user_batches(d["users"]), score_batch, d)
