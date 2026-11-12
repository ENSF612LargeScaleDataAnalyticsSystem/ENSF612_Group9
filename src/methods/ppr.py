"""Monte Carlo personalised PageRank on the bipartite user-movie like graph.

Edges join a user to every movie they liked. For a user u, n_walkers walkers start at u. At each
of n_steps steps a walker moves user -> random liked movie (the visit is counted) -> random user
who liked that movie, and then jumps back to u with probability alpha (the restart). The visit
counts estimate u's PPR scores over movies. With beta > 0 the counts are divided by
(likes of the movie)^beta, which lowers the weight of very popular movies (as in RP3beta).

Walks of a whole batch of users are simulated together with numpy. The random generator is seeded
by (seed, first user of the batch) and batches are fixed, so results do not depend on the cores.
Users with no like cannot start a walk and get the popular items.

Visit counts are a Monte Carlo estimate, so a movie reached once or twice has a very noisy score; with
beta > 0 such rare movies get the largest boost and crowd out the rest (on the pilot, beta 0.5 with 100
walkers gives NDCG@10 ~0). min_visits drops movies visited fewer times than that before scoring.

Params: alpha (0.3), n_walkers (100), n_steps (20), beta (0.0), min_visits (1), seed (common.SEED).
"""
import numpy as np
from pyspark.sql import DataFrame, SparkSession

from src.common import SEED
from src.methods._shared import build_index, ranked_rows, row, run_batches, top_k, user_batches


def walk_counts(batch: np.ndarray, d: dict, alpha: float, n_walkers: int, n_steps: int,
                rng: np.random.Generator) -> np.ndarray:
    """(len(batch), n_items) visit counts."""
    n_items = d["n_items"]
    lp, li = d["liked_ptr"], d["liked_idx"]
    kp, ki = d["likers_ptr"], d["likers_idx"]
    deg = lp[batch + 1] - lp[batch]
    counts = np.zeros(len(batch) * n_items, np.float64)
    active = np.flatnonzero(deg > 0)
    if len(active) == 0:
        return counts.reshape(len(batch), n_items)
    origin_row = np.repeat(active, n_walkers)            # row in `counts` of each walker
    origin = batch[origin_row].astype(np.int64)
    cur = origin.copy()
    for _ in range(n_steps):
        d_u = lp[cur + 1] - lp[cur]
        item = li[lp[cur] + (rng.random(len(cur)) * d_u).astype(np.int64)]
        counts += np.bincount(origin_row * n_items + item, minlength=len(counts))
        d_i = kp[item + 1] - kp[item]
        nxt = ki[kp[item] + (rng.random(len(cur)) * d_i).astype(np.int64)].astype(np.int64)
        cur = np.where(rng.random(len(cur)) < alpha, origin, nxt)
    return counts.reshape(len(batch), n_items)


def recommend(spark: SparkSession, train: DataFrame, k: int = 10, alpha: float = 0.3,
              n_walkers: int = 100, n_steps: int = 20, beta: float = 0.0, min_visits: int = 1,
              seed: int = SEED) -> DataFrame:
    d = build_index(train)
    alpha, n_walkers, n_steps, beta, seed = float(alpha), int(n_walkers), int(n_steps), float(beta), int(seed)
    min_visits = int(min_visits)
    if beta:
        d["penalty"] = np.maximum(d["n_likes"], 1).astype(np.float64) ** -beta

    def score_batch(batch, d):
        rng = np.random.default_rng([seed, int(batch[0])])
        counts = walk_counts(batch, d, alpha, n_walkers, n_steps, rng)
        if min_visits > 1:
            counts[counts < min_visits] = 0          # too few visits: estimate is noise
        if beta:
            counts *= d["penalty"][None, :]
        out = []
        for j, u in enumerate(batch):
            seen = row(d["rated_ptr"], d["rated_idx"], u)
            out.extend(ranked_rows(u, top_k(counts[j], seen, k, d["pop_order"])))
        return out

    return run_batches(spark, user_batches(d["users"]), score_batch, d)
