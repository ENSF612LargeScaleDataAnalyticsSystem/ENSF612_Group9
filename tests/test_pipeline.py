"""Small hand-checked cases for the shared code. Run from the project root: python -m pytest -q"""
import math

import pytest
from pyspark.sql import SparkSession

from src.common import use_this_python
from src.evaluate import evaluate, per_user_metrics
from src.methods import popularity
from src.split import global_split, per_user_split

RATINGS = "user INT, item INT, rating DOUBLE, ts LONG, like BOOLEAN"


@pytest.fixture(scope="session")
def spark():
    use_this_python()
    s = (SparkSession.builder.master("local[1]").appName("tests")
         .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false").getOrCreate())
    yield s
    s.stop()


def test_metrics_by_hand(spark):
    # k = 3. User 1 likes {10, 20}, user 2 likes {30}, user 3 likes {40} and gets no list.
    truth = spark.createDataFrame([(1, 10), (1, 20), (2, 30), (3, 40)], "user INT, item INT")
    recs = spark.createDataFrame([(1, 10, 1), (1, 99, 2), (1, 20, 3), (2, 31, 1), (2, 32, 2), (2, 30, 3)],
                                 "user INT, item INT, rank INT")
    pu = {r["user"]: r for r in per_user_metrics(recs, truth, k=3).collect()}
    ndcg1 = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert pu[1]["precision"] == pytest.approx(2 / 3) and pu[1]["recall"] == pytest.approx(1.0)
    assert pu[1]["ndcg"] == pytest.approx(ndcg1)
    assert pu[2]["ndcg"] == pytest.approx(0.5) and pu[2]["recall"] == pytest.approx(1.0)
    assert pu[3]["hits"] == 0 and pu[3]["ndcg"] == 0

    m = evaluate(recs, truth, k=3, n_boot=200)
    assert m["evaluated_users"] == 3 and m["evaluated_users_without_list"] == 1
    assert m["precision@3"] == pytest.approx(1 / 3, abs=1e-6)
    assert m["recall@3"] == pytest.approx(2 / 3, abs=1e-6)
    assert m["ndcg@3"] == pytest.approx((ndcg1 + 0.5) / 3, abs=1e-6)
    lo, hi = m["ndcg@3_ci95"]
    assert lo <= m["ndcg@3"] <= hi


def test_bad_recommendations_are_rejected(spark):
    truth = spark.createDataFrame([(1, 10)], "user INT, item INT")
    dup = spark.createDataFrame([(1, 10, 1), (1, 10, 2)], "user INT, item INT, rank INT")
    with pytest.raises(ValueError):
        evaluate(dup, truth, k=10, n_boot=0)
    bad_rank = spark.createDataFrame([(1, 10, 11)], "user INT, item INT, rank INT")
    with pytest.raises(ValueError):
        evaluate(bad_rank, truth, k=10, n_boot=0)


def test_seen_items_are_counted(spark):
    train = spark.createDataFrame([(1, 5, 4.0, 1, True)], RATINGS)
    truth = spark.createDataFrame([(1, 10)], "user INT, item INT")
    recs = spark.createDataFrame([(1, 5, 1), (1, 10, 2)], "user INT, item INT, rank INT")
    assert evaluate(recs, truth, train=train, k=10, n_boot=0)["recommended_items_already_rated"] == 1


def test_split_sizes_and_ties(spark):
    # 35 ratings: test = ceil(7.0) = 7 (not 8), val = ceil(3.5) = 4. All share one timestamp,
    # so the order falls back to item id and the result is deterministic.
    rows = [(1, i, 4.0, 100, True) for i in range(35)]
    parts = {r["item"]: r["part"] for r in per_user_split(spark.createDataFrame(rows, RATINGS)).collect()}
    assert [i for i in range(35) if parts[i] == "test"] == list(range(28, 35))
    assert [i for i in range(35) if parts[i] == "val"] == list(range(24, 28))


def test_global_cutoffs_are_exact_and_partition_independent(spark):
    # 100 ratings with timestamps 0..99: validation must start at ts 70 and test at ts 80,
    # however the data is partitioned.
    rows = [(u % 7, u, 4.0, u, True) for u in range(100)]
    df = spark.createDataFrame(rows, RATINGS)
    for parts in (1, 3, 8):
        _, cut = global_split(df.repartition(parts))
        assert cut == {"val_from_ts": 70, "test_from_ts": 80}


def test_popularity_skips_seen_items(spark):
    # Item 1 is the most liked, then 2, then 3. User 9 already rated item 1.
    rows = [(u, 1, 5.0, 1, True) for u in range(5)] + [(u, 2, 4.0, 1, True) for u in range(3)] + \
           [(0, 3, 4.5, 1, True), (9, 1, 2.0, 1, False)]
    recs = popularity.recommend(spark, spark.createDataFrame(rows, RATINGS), k=2)
    got = {(r["user"], r["rank"]): r["item"] for r in recs.collect()}
    assert got[(9, 1)] == 2 and got[(9, 2)] == 3
    assert (0, 1) not in got or got[(0, 1)] not in (1, 2, 3)  # user 0 has rated every item
