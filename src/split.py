"""Step 3 - splits: train / validation / test, plus a 10% pilot sample of each.

    python -m src.split

Two splits (proposal, Section 3):
    user_time    For each user, sort ratings by (timestamp, item). The last ceil(20% of n) ratings
                 are test, the ceil(10% of n) before them validation, the rest train. Sorting by
                 item as a second key makes ties (many ratings in one session) deterministic.
    global_time  One cut-off date for everybody: the latest 20% of all ratings are test, the 10%
                 before them validation. Users who first appear after a cut-off have no history
                 at that stage and are not evaluated (see evaluate.ground_truth).

Pilot samples user_time_10pct and global_time_10pct keep 10% of users (common.user_sample),
with all of their ratings. Use them while developing; a full run can take much longer.

Outputs: data/splits/<split>/{train,val,test} and reports/split_stats.{json,md}
"""
import argparse
import json
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.common import PILOT_FRAC, PROCESSED, REPORTS, SPLITS, TEST_PCT, VAL_PCT, get_spark, user_sample

PARTS = ("train", "val", "test")


def per_user_split(ratings: DataFrame, test_pct: int = TEST_PCT, val_pct: int = VAL_PCT) -> DataFrame:
    """Add a column `part` = train / val / test using each user's own time order."""
    order = Window.partitionBy("user").orderBy("ts", "item")
    whole = Window.partitionBy("user")
    df = (ratings.withColumn("pos", F.row_number().over(order))
          .withColumn("n", F.count("*").over(whole))
          # integer ceil(n * pct / 100); plain float math can turn 7.0 into 7.000000000000001
          .withColumn("n_test", F.expr(f"(n * {test_pct} + 99) div 100"))
          .withColumn("n_val", F.expr(f"(n * {val_pct} + 99) div 100")))
    part = (F.when(F.col("pos") > F.col("n") - F.col("n_test"), "test")
            .when(F.col("pos") > F.col("n") - F.col("n_test") - F.col("n_val"), "val")
            .otherwise("train"))
    return df.withColumn("part", part).drop("pos", "n", "n_test", "n_val")


def global_split(ratings: DataFrame, test_pct: int = TEST_PCT, val_pct: int = VAL_PCT):
    """Add `part` using two cut-off timestamps shared by all users. Returns (df, cutoffs).

    The cut-offs are exact order statistics of the timestamps: the (k+1)-th smallest timestamp,
    with k = 70% and 80% of all ratings. approxQuantile would depend on how the data is
    partitioned, so two group members with different core counts could get different splits.
    """
    n = ratings.count()
    k_val = n * (100 - test_pct - val_pct) // 100
    k_test = n * (100 - test_pct) // 100
    ts_sorted = ratings.select("ts").orderBy("ts")
    t_val = int(ts_sorted.offset(k_val).limit(1).first()[0])
    t_test = int(ts_sorted.offset(k_test).limit(1).first()[0])
    part = (F.when(F.col("ts") >= t_test, "test").when(F.col("ts") >= t_val, "val").otherwise("train"))
    return ratings.withColumn("part", part), {"val_from_ts": t_val, "test_from_ts": t_test}


def write_parts(df: DataFrame, out: Path):
    df = df.persist(StorageLevel.MEMORY_AND_DISK)
    for p in PARTS:
        df.where(F.col("part") == p).drop("part").write.mode("overwrite").parquet(str(out / p))
    df.unpersist()


def stats(spark, d: Path) -> dict:
    parts = {p: spark.read.parquet(str(d / p)) for p in PARTS}
    s = {}
    for p, df in parts.items():
        row = df.agg(F.count("*").alias("ratings"), F.countDistinct("user").alias("users"),
                     F.countDistinct("item").alias("items"),
                     F.sum(F.col("like").cast("int")).alias("likes")).first().asDict()
        s[p] = {k: int(v or 0) for k, v in row.items()}
    total = sum(s[p]["ratings"] for p in PARTS)
    for p in PARTS:
        s[p]["share_pct"] = round(100 * s[p]["ratings"] / total, 2)

    # What each stage will evaluate: users with >= 1 liked target item and some training history.
    for stage, train, target in [("val_stage", parts["train"], parts["val"]),
                                 ("test_stage", parts["train"].unionByName(parts["val"]), parts["test"])]:
        train_users = train.select("user").distinct()
        train_items = train.select("item").distinct()
        liked = target.where("like")
        s[stage] = {
            "evaluable_users": liked.join(train_users, "user", "left_semi").select("user").distinct().count(),
            "target_users_without_training_history":
                target.select("user").distinct().join(train_users, "user", "left_anti").count(),
            "target_likes_on_items_unseen_in_training":
                liked.join(train_items, "item", "left_anti").count(),
            "target_likes": liked.count(),
        }
    return s


def run(spark, processed: Path, splits: Path, reports: Path, pilot_frac: float):
    t0 = time.time()
    ratings = spark.read.parquet(str(processed / "ratings"))
    all_stats = {}

    write_parts(per_user_split(ratings), splits / "user_time")
    g, cutoffs = global_split(ratings)
    write_parts(g, splits / "global_time")

    pct = int(round(pilot_frac * 100))
    for name in ["user_time", "global_time"]:
        for p in PARTS:
            df = spark.read.parquet(str(splits / name / p))
            user_sample(df, pilot_frac).write.mode("overwrite").parquet(str(splits / f"{name}_{pct}pct" / p))
        for n in [name, f"{name}_{pct}pct"]:
            all_stats[n] = stats(spark, splits / n)

    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t))
    all_stats["global_time"]["cutoffs"] = {**cutoffs, "val_from_utc": fmt(cutoffs["val_from_ts"]),
                                           "test_from_utc": fmt(cutoffs["test_from_ts"])}
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "split_stats.json").write_text(json.dumps(all_stats, indent=2))
    write_markdown(reports / "split_stats.md", all_stats)
    print(f"splits done in {time.time() - t0:.0f}s -> {splits}")
    for n, s in all_stats.items():
        print(f"  {n:<18} train {s['train']['ratings']:>11,}  val {s['val']['ratings']:>10,}  "
              f"test {s['test']['ratings']:>10,}  evaluable users (test stage) {s['test_stage']['evaluable_users']:,}")


def write_markdown(path: Path, all_stats: dict):
    lines = ["# Split statistics (generated by src/split.py)", ""]
    for n, s in all_stats.items():
        lines += [f"## {n}", "", "| Part | Ratings | Share | Users | Movies | Likes |", "|---|---|---|---|---|---|"]
        for p in PARTS:
            x = s[p]
            lines.append(f"| {p} | {x['ratings']:,} | {x['share_pct']}% | {x['users']:,} | {x['items']:,} | {x['likes']:,} |")
        lines += ["", "| Stage | Evaluable users | Target users without history | Target likes on items unseen in training |",
                  "|---|---|---|---|"]
        for st in ["val_stage", "test_stage"]:
            x = s[st]
            lines.append(f"| {st} | {x['evaluable_users']:,} | {x['target_users_without_training_history']:,} | "
                         f"{x['target_likes_on_items_unseen_in_training']:,} of {x['target_likes']:,} |")
        if "cutoffs" in s:
            c = s["cutoffs"]
            lines += ["", f"Validation from {c['val_from_utc']} UTC, test from {c['test_from_utc']} UTC."]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--out", type=Path, default=SPLITS)
    ap.add_argument("--reports", type=Path, default=REPORTS)
    ap.add_argument("--pilot-frac", type=float, default=PILOT_FRAC)
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()
    spark = get_spark("split", driver_memory=args.driver_memory)
    try:
        run(spark, args.processed, args.out, args.reports, args.pilot_frac)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
