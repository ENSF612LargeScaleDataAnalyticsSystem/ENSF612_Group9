"""Helpers shared by the neighbourhood, random-walk and ALS methods.

The training data (14M ratings at most) is turned into compressed sparse rows (CSR) with numpy,
broadcast once to every worker, and users are scored in fixed batches. The batches depend only on
the user ids, not on the number of cores, so results are identical for --cores 1, 2 and 4.

Index built from train (all arrays numpy):
    rated_ptr/rated_idx    user -> every item the user rated (to never recommend a seen item)
    liked_ptr/liked_idx    user -> items the user liked
    likers_ptr/likers_idx  item -> users who liked it
    pop_order              items ordered like the popularity baseline (likes, ratings, id)
Items inside each CSR row are sorted.
"""
import numpy as np
from pyspark.sql import DataFrame, SparkSession

BATCH = 256          # users per batch; fixed so results do not depend on partitioning
REC_SCHEMA = "user INT, item INT, rank INT"


def _pack(rows):
    u, i, l = [], [], []
    for r in rows:
        u.append(r[0]); i.append(r[1]); l.append(r[2])
    yield (np.asarray(u, np.int32).tobytes(), np.asarray(i, np.int32).tobytes(),
           np.asarray(l, np.bool_).tobytes())


def to_numpy(train: DataFrame):
    """(user, item, like) as three numpy arrays. Packed per partition, so the driver gets bytes."""
    parts = train.select("user", "item", "like").rdd.mapPartitions(_pack).collect()
    users = np.concatenate([np.frombuffer(p[0], np.int32) for p in parts])
    items = np.concatenate([np.frombuffer(p[1], np.int32) for p in parts])
    likes = np.concatenate([np.frombuffer(p[2], np.bool_) for p in parts])
    return users, items, likes


def csr(rows: np.ndarray, cols: np.ndarray, n_rows: int):
    """CSR (ptr, idx) with the column ids of each row sorted."""
    order = np.lexsort((cols, rows))
    idx = cols[order].astype(np.int32)
    ptr = np.zeros(n_rows + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=n_rows), out=ptr[1:])
    return ptr, idx


def build_index(train: DataFrame) -> dict:
    users, items, likes = to_numpy(train)
    n_users = int(users.max()) + 1
    n_items = int(items.max()) + 1
    rated_ptr, rated_idx = csr(users, items, n_users)
    liked_ptr, liked_idx = csr(users[likes], items[likes], n_users)
    likers_ptr, likers_idx = csr(items[likes], users[likes], n_items)
    n_likes = np.bincount(items[likes], minlength=n_items)
    n_ratings = np.bincount(items, minlength=n_items)
    item_ids = np.arange(n_items)
    in_train = n_ratings > 0
    order = np.lexsort((item_ids, -n_ratings, -n_likes))      # same order as popularity.py
    pop_order = order[in_train[order]].astype(np.int32)
    return {
        "n_users": n_users, "n_items": n_items,
        "users": np.unique(users).astype(np.int32),
        "rated_ptr": rated_ptr, "rated_idx": rated_idx,
        "liked_ptr": liked_ptr, "liked_idx": liked_idx,
        "likers_ptr": likers_ptr, "likers_idx": likers_idx,
        "pop_order": pop_order, "n_likes": n_likes,
    }


def row(ptr, idx, r):
    return idx[ptr[r]:ptr[r + 1]]


def gather(ptr, idx, rows):
    """Concatenate CSR rows `rows`; also return the row lengths."""
    starts, ends = ptr[rows], ptr[rows + 1]
    lens = ends - starts
    if len(rows) == 0 or lens.sum() == 0:
        return np.empty(0, idx.dtype), lens
    # positions starts[j] .. ends[j]-1 for every j, without a Python loop
    offs = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
    return idx[np.arange(lens.sum()) + offs], lens


def best_idx(values: np.ndarray, ids: np.ndarray, n: int) -> np.ndarray:
    """Positions of the n largest values, ordered by (value desc, id asc).

    Exact in the presence of ties: argpartition alone picks an arbitrary member of a tie at the
    cut-off, which depends on the input order (and so on Spark's partitioning).
    """
    if len(values) > n:
        thr = np.partition(values, len(values) - n)[len(values) - n]   # n-th largest value
        above = np.flatnonzero(values > thr)
        tied = np.flatnonzero(values == thr)
        tied = tied[np.argsort(ids[tied], kind="stable")][:n - len(above)]
        pos = np.concatenate([above, tied])
    else:
        pos = np.arange(len(values))
    return pos[np.lexsort((ids[pos], -values[pos]))]


def top_k(scores: np.ndarray, seen: np.ndarray, k: int, pop_order: np.ndarray, valid=None):
    """k best items by score (ties: lower item id), never a seen item.

    Only items with score > 0 count, unless `valid` (bool mask) says which items are candidates.
    If fewer than k items qualify, the list is filled with the most popular unseen items, so every
    user gets k items (an empty list would score 0 anyway).
    """
    s = scores.astype(np.float64, copy=True)
    ok = (s > 0) if valid is None else valid.copy()
    ok[seen] = False
    cand = np.flatnonzero(ok)
    chosen = cand[best_idx(s[cand], cand, k)]
    if len(chosen) < k:
        blocked = np.zeros(len(s), np.bool_)
        blocked[seen] = True
        blocked[chosen] = True
        fill = pop_order[~blocked[pop_order]]
        chosen = np.concatenate([chosen, fill[:k - len(chosen)]])
    return chosen[:k]


def user_batches(users: np.ndarray):
    users = np.sort(users)
    return [users[i:i + BATCH] for i in range(0, len(users), BATCH)]


def run_batches(spark: SparkSession, batches, fn, broadcast_value) -> DataFrame:
    """Apply fn(batch, data) -> iterable of (user, item, rank) to every batch, in parallel."""
    sc = spark.sparkContext
    bc = sc.broadcast(broadcast_value)
    n_slices = max(1, min(len(batches), sc.defaultParallelism * 4))

    def work(it):
        data = bc.value
        for batch in it:
            yield from fn(batch, data)

    rdd = sc.parallelize(batches, n_slices).mapPartitions(work)
    return spark.createDataFrame(rdd, REC_SCHEMA)


def ranked_rows(user, items):
    return [(int(user), int(it), r + 1) for r, it in enumerate(items)]


def neighbour_scores(nbrs: np.ndarray, sims: np.ndarray, d: dict) -> np.ndarray:
    """score(item) = sum of similarities of the neighbours who liked the item."""
    items, lens = gather(d["liked_ptr"], d["liked_idx"], nbrs)
    return np.bincount(items, weights=np.repeat(sims, lens), minlength=d["n_items"])


def similarity(inter: np.ndarray, deg_u: int, deg_v: np.ndarray, kind: str) -> np.ndarray:
    inter = inter.astype(np.float64)
    if kind == "jaccard":
        return inter / (deg_u + deg_v - inter)
    if kind == "cosine":
        return inter / np.sqrt(float(deg_u) * deg_v)
    raise ValueError("similarity must be 'jaccard' or 'cosine'")


def best_n(nbrs: np.ndarray, sims: np.ndarray, n: int):
    """n most similar neighbours (ties: lower user id)."""
    pos = best_idx(sims, nbrs, n)
    return nbrs[pos], sims[pos]
