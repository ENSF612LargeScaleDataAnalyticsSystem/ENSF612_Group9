"""Checks for the four methods and their shared helpers. Run from the project root: python -m pytest -q"""
import random

import numpy as np
import pytest
from pyspark.sql import SparkSession

from src.common import use_this_python
from src.evaluate import check_recs
from src.methods import als, knn, lsh, popularity, ppr
from src.methods._shared import best_idx, build_index, gather, top_k

RATINGS = "user INT, item INT, rating DOUBLE, ts LONG, like BOOLEAN"


@pytest.fixture(scope="module")
def spark():
    use_this_python()
    s = (SparkSession.builder.master("local[2]").appName("tests-methods")
         .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def ratings(spark, triples):
    """(user, item, rating) -> train DataFrame in the project schema."""
    return spark.createDataFrame([(u, i, float(r), 1000 + n, r >= 4.0) for n, (u, i, r) in enumerate(triples)],
                                 RATINGS)


@pytest.fixture(scope="module")
def random_train(spark):
    rng = random.Random(7)
    rows = {(u, rng.randrange(60)): rng.choice([1.0, 2.5, 3.0, 4.0, 4.5, 5.0]) for u in range(80) for _ in range(25)}
    return ratings(spark, [(u, i, r) for (u, i), r in rows.items()]).cache()


def lists(recs):
    out = {}
    for r in recs.orderBy("user", "rank").collect():
        out.setdefault(r["user"], []).append(r["item"])
    return out


def test_gather_matches_a_loop():
    ptr = np.array([0, 2, 2, 5, 6])
    idx = np.array([10, 11, 12, 13, 14, 15])
    got, lens = gather(ptr, idx, np.array([2, 0, 1, 3]))
    assert got.tolist() == [12, 13, 14, 10, 11, 15]
    assert lens.tolist() == [3, 2, 0, 1]


def test_top_k_skips_seen_breaks_ties_and_fills():
    scores = np.array([0.0, 5.0, 5.0, 3.0, 9.0, 0.0])
    pop = np.array([5, 0, 4, 1, 2, 3])
    # item 4 is seen; 1 and 2 tie on 5.0 -> lower id first; only 3 items have a score -> fill with 5
    assert top_k(scores, np.array([4]), 4, pop).tolist() == [1, 2, 3, 5]


def test_best_idx_breaks_ties_by_id_whatever_the_input_order():
    # values tie at the cut-off (0.5 for ids 3, 7, 9); n = 3 -> 0.9 (id 5), then ids 3 and 7
    ids = np.array([9, 5, 7, 3, 1])
    vals = np.array([0.5, 0.9, 0.5, 0.5, 0.1])
    for perm in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 0, 4, 1, 3]):
        got = ids[perm][best_idx(vals[perm], ids[perm], 3)]
        assert got.tolist() == [5, 3, 7]


def test_knn_by_hand(spark):
    # likes: u0 {0,1}, u1 {0,1,2}, u2 {3}; u0 also rated item 3 (not liked).
    # u0's only neighbour is u1 (Jaccard 2/3); its one unseen liked item is 2.
    train = ratings(spark, [(0, 0, 5), (0, 1, 4), (0, 3, 2), (1, 0, 5), (1, 1, 5), (1, 2, 4), (2, 3, 5)])
    recs = lists(knn.recommend(spark, train, k=1, n_neighbours=5))
    assert recs[0] == [2]
    # u2 shares no like with anyone -> popularity fill: most-liked unseen item is 0 (2 likes, id 0 < 1)
    assert recs[2] == [0]


def test_lsh_with_many_bands_finds_the_exact_neighbours(spark, random_train):
    # r = 1 row per band and 128 bands: any pair with Jaccard >= 0.1 is a candidate with
    # probability 1 - 0.9^128 > 0.99999, so LSH should give the exact kNN lists.
    exact = lists(knn.recommend(spark, random_train, k=5, n_neighbours=10))
    approx = lists(lsh.recommend(spark, random_train, k=5, n_hashes=128, bands=128, n_neighbours=10,
                                 max_bucket=10_000))
    same = sum(exact[u] == approx[u] for u in exact)
    assert same >= 0.95 * len(exact)


@pytest.mark.parametrize("name,params", [
    ("popularity", {}), ("knn", {}), ("lsh", {"n_hashes": 32, "bands": 16}),
    ("ppr", {"n_walkers": 20, "n_steps": 5}), ("als", {"rank": 8, "max_iter": 3}),
])
def test_method_contract(spark, random_train, name, params):
    mod = {"popularity": popularity, "knn": knn, "lsh": lsh, "ppr": ppr, "als": als}[name]
    k = 5
    recs = mod.recommend(spark, random_train, k=k, **params).cache()
    info = check_recs(recs, k, train=random_train)               # raises on duplicates / bad ranks
    assert info["recommended_items_already_rated"] == 0
    assert info["rec_users"] == random_train.select("user").distinct().count()
    assert info["rec_rows"] == k * info["rec_users"]
    if name != "als":                                            # ALS is deterministic, but slow to rerun
        assert lists(recs) == lists(mod.recommend(spark, random_train, k=k, **params))


def test_ppr_restart_one_only_reaches_own_neighbourhood(spark):
    # alpha = 1: every walk is user -> liked item -> back, so u0 only visits its own (seen) items
    # and gets the popularity fill; u1 likes 0 and 2 -> nothing unseen either.
    train = ratings(spark, [(0, 0, 5), (0, 1, 5), (1, 0, 5), (1, 2, 5), (2, 3, 5)])
    d = build_index(train)
    counts = ppr.walk_counts(np.array([0], np.int32), d, alpha=1.0, n_walkers=50, n_steps=3,
                             rng=np.random.default_rng(0))
    assert set(np.flatnonzero(counts[0])) <= {0, 1}
