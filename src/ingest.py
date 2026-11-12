"""Step 2 - ingest: raw CSV -> checked, cleaned Parquet with dense ids, plus a data profile.

    python -m src.ingest                       # reads data/raw/ml-20m
    python -m src.ingest --raw data/raw/synthetic

Outputs
    data/processed/ratings   user, item, rating, ts, like   (dense ids, one row per user-movie pair)
    data/processed/users     userId -> user
    data/processed/movies    movieId -> item (null if the movie has no ratings), title, genres, counts
    reports/data_quality.json, reports/data_profile.json, reports/data_profile.md
    reports/rating_distribution.csv, ratings_per_year.csv, ratings_per_user.csv, ratings_per_movie.csv

Cleaning rules (every one is counted in reports/data_quality.json, even when the count is 0):
    1. Drop rating rows with a missing or unparseable field.
    2. Drop ratings outside 0.5-5.0 or not on a half-star step.
    3. Drop ratings with a timestamp outside the collection period (TS_MIN..TS_MAX).
    4. Drop ratings whose movieId is not in movies.csv.
    5. If a user rated the same movie more than once, keep the latest rating.
    Movies without any rating get no dense id: no collaborative method can recommend them.
    Users and movies are never removed for being unusual (e.g. very many ratings).
"""
import argparse
import csv
import json
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import functions as F

from src.common import LIKE_THRESHOLD, PROCESSED, RAW, REPORTS, TS_MAX, TS_MIN, get_spark


def try_cast(col: str, typ: str):
    # Spark 4 runs in ANSI mode, where a plain cast of a bad string raises an error.
    return F.expr(f"try_cast(`{col}` AS {typ})")


def blank(col: str):
    return F.col(col).isNull() | (F.trim(F.col(col)) == "")


def flag_sum(cond, name):
    return F.sum(F.when(cond, 1).otherwise(0)).alias(name)


def read_ratings(spark, raw: Path):
    df = spark.read.option("header", True).csv(str(raw / "ratings.csv"))
    expected = ["userId", "movieId", "rating", "timestamp"]
    if df.columns != expected:
        raise ValueError(f"ratings.csv columns are {df.columns}, expected {expected}")
    return df.select(
        F.col("userId").alias("userId_s"), F.col("movieId").alias("movieId_s"),
        F.col("rating").alias("rating_s"), F.col("timestamp").alias("ts_s"),
        try_cast("userId", "INT").alias("userId"), try_cast("movieId", "INT").alias("movieId"),
        try_cast("rating", "DOUBLE").alias("rating"), try_cast("timestamp", "BIGINT").alias("ts"),
    )


def read_movies(spark, raw: Path):
    # Titles contain commas and doubled quotes ("") inside quoted fields.
    df = (spark.read.option("header", True).option("quote", '"').option("escape", '"')
          .csv(str(raw / "movies.csv")))
    expected = ["movieId", "title", "genres"]
    if df.columns != expected:
        raise ValueError(f"movies.csv columns are {df.columns}, expected {expected}")
    return df.select(F.col("movieId").alias("movieId_s"), try_cast("movieId", "INT").alias("movieId"),
                     "title", "genres")


def quantiles(df, col, probs=(0.25, 0.5, 0.75, 0.9, 0.99)):
    row = df.agg(
        F.min(col).alias("min"), F.mean(col).alias("mean"), F.max(col).alias("max"),
        F.percentile_approx(col, list(probs), 100000).alias("q"),
    ).first()
    out = {"min": row["min"], "mean": round(float(row["mean"]), 2), "max": row["max"]}
    out.update({f"p{int(p * 100)}": v for p, v in zip(probs, row["q"])})
    return out


def write_csv(path: Path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def run(spark, raw: Path, out: Path, reports: Path):
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    q = {}  # data-quality counts

    # ------------------------------------------------------------ ratings: field checks
    r = read_ratings(spark, raw)
    valid_rating = (F.col("rating") >= 0.5) & (F.col("rating") <= 5.0) & \
                   (F.col("rating") * 2 == F.floor(F.col("rating") * 2))
    ts_ok = F.col("ts").between(TS_MIN, TS_MAX)
    row = r.agg(
        F.count("*").alias("rows_raw"),
        *[flag_sum(blank(f"{c}_s"), f"missing_{c}") for c in ["userId", "movieId", "rating", "ts"]],
        *[flag_sum(~blank(f"{c}_s") & F.col(c).isNull(), f"unparseable_{c}")
          for c in ["userId", "movieId", "rating", "ts"]],
        flag_sum(F.col("rating").isNotNull() & ~valid_rating, "invalid_rating_value"),
        flag_sum(F.col("ts").isNotNull() & ~ts_ok, "timestamp_out_of_range"),
    ).first()
    q.update(row.asDict())

    typed_ok = F.col("userId").isNotNull() & F.col("movieId").isNotNull() & \
        F.col("rating").isNotNull() & F.col("ts").isNotNull()
    valid = r.where(typed_ok & valid_rating & ts_ok).select("userId", "movieId", "rating", "ts")

    # ------------------------------------------------------------ movies: checks
    m = read_movies(spark, raw)
    mrow = m.agg(
        F.count("*").alias("movies_rows_raw"),
        flag_sum(F.col("movieId").isNull(), "movies_bad_movieId"),
        flag_sum(~F.coalesce(F.col("title"), F.lit("")).rlike(r"\(\d{4}\)\s*$"), "movies_title_without_year"),
        flag_sum(F.coalesce(F.col("genres"), F.lit("(no genres listed)")) == "(no genres listed)",
                 "movies_without_genres"),
    ).first()
    q.update(mrow.asDict())
    movies = (m.where(F.col("movieId").isNotNull())
              .groupBy("movieId").agg(F.min("title").alias("title"), F.min("genres").alias("genres"),
                                      F.count("*").alias("_n")))
    q["movies_duplicate_movieId"] = movies.where(F.col("_n") > 1).count()
    movies = movies.drop("_n")
    n_movies_csv = movies.count()

    # ------------------------------------------------------------ orphans and duplicates
    q["ratings_orphan_movieId"] = valid.join(F.broadcast(movies.select("movieId")), "movieId", "left_anti").count()
    valid = valid.join(F.broadcast(movies.select("movieId")), "movieId", "left_semi")

    grouped = (valid.groupBy("userId", "movieId")
               .agg(F.count("*").alias("n"),
                    F.max(F.struct("ts", "rating")).alias("latest"),
                    (F.min("rating") != F.max("rating")).alias("conflict"))
               .persist(StorageLevel.MEMORY_AND_DISK))
    drow = grouped.agg(
        flag_sum(F.col("n") > 1, "duplicate_pairs"),
        F.sum(F.col("n") - 1).alias("duplicate_extra_rows"),
        flag_sum((F.col("n") > 1) & F.col("conflict"), "duplicate_pairs_with_different_ratings"),
    ).first()
    q.update({k: int(v or 0) for k, v in drow.asDict().items()})
    clean = grouped.select("userId", "movieId", F.col("latest.rating").alias("rating"),
                           F.col("latest.ts").alias("ts"))

    # ------------------------------------------------------------ dense ids
    # Sorted original ids -> 0..N-1. Collected to the driver: only ~138k users and ~27k movies.
    user_ids = sorted(x[0] for x in clean.select("userId").distinct().collect())
    item_ids = sorted(x[0] for x in clean.select("movieId").distinct().collect())
    users = spark.createDataFrame([(int(u), i) for i, u in enumerate(user_ids)], "userId INT, user INT")
    items = spark.createDataFrame([(int(mv), i) for i, mv in enumerate(item_ids)], "movieId INT, item INT")

    ratings = (clean.join(F.broadcast(users), "userId").join(F.broadcast(items), "movieId")
               .select("user", "item", "rating", "ts", (F.col("rating") >= LIKE_THRESHOLD).alias("like")))
    ratings.write.mode("overwrite").parquet(str(out / "ratings"))
    grouped.unpersist()
    users.write.mode("overwrite").parquet(str(out / "users"))

    ratings = spark.read.parquet(str(out / "ratings"))
    per_item = ratings.groupBy("item").agg(F.count("*").alias("n_ratings"),
                                           F.sum(F.col("like").cast("int")).alias("n_likes"))
    movies_out = (movies.join(items, "movieId", "left").join(per_item, "item", "left")
                  .select("movieId", "item", "title", "genres",
                          F.coalesce("n_ratings", F.lit(0)).alias("n_ratings"),
                          F.coalesce("n_likes", F.lit(0)).alias("n_likes")))
    movies_out.write.mode("overwrite").parquet(str(out / "movies"))
    q["movies_without_ratings"] = movies_out.where(F.col("item").isNull()).count()
    q["rows_clean"] = ratings.count()
    q["rows_dropped_total"] = q["rows_raw"] - q["rows_clean"]

    # ------------------------------------------------------------ profile
    p = {}
    n_r, n_u, n_i = q["rows_clean"], len(user_ids), len(item_ids)
    p["ratings"], p["users"], p["movies_with_ratings"] = n_r, n_u, n_i
    p["movies_in_movies_csv"] = n_movies_csv
    p["density_pct"] = round(100 * n_r / (n_u * n_i), 4)
    n_likes = ratings.where("like").count()
    p["likes"], p["like_share_pct"] = n_likes, round(100 * n_likes / n_r, 2)
    tsr = ratings.agg(F.min("ts"), F.max("ts")).first()
    p["first_rating_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(tsr[0]))
    p["last_rating_utc"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(tsr[1]))

    per_user = ratings.groupBy("user").agg(F.count("*").alias("n_ratings"),
                                           F.sum(F.col("like").cast("int")).alias("n_likes"))
    p["ratings_per_user"] = quantiles(per_user, "n_ratings")
    p["likes_per_user"] = quantiles(per_user, "n_likes")
    p["ratings_per_movie"] = quantiles(per_item, "n_ratings")
    p["users_with_fewer_than_20_ratings"] = per_user.where(F.col("n_ratings") < 20).count()
    p["users_without_any_like"] = per_user.where(F.col("n_likes") == 0).count()
    p["movies_without_any_like"] = per_item.where(F.col("n_likes") == 0).count()
    p["movies_with_1_rating"] = per_item.where(F.col("n_ratings") == 1).count()

    top1 = max(1, n_i // 100)
    top_sum = per_item.orderBy(F.desc("n_ratings")).limit(top1).agg(F.sum("n_ratings")).first()[0]
    p["share_of_ratings_on_top_1pct_movies_pct"] = round(100 * top_sum / n_r, 2)
    same_ts = (ratings.groupBy("user", "ts").count().where(F.col("count") > 1)
               .agg(F.sum("count")).first()[0] or 0)
    p["ratings_sharing_a_timestamp_with_same_user_pct"] = round(100 * same_ts / n_r, 2)

    dist = ratings.groupBy("rating").count().orderBy("rating").collect()
    write_csv(reports / "rating_distribution.csv", ["rating", "count"], [(x[0], x[1]) for x in dist])
    years = (ratings.groupBy(F.year(F.timestamp_seconds("ts")).alias("year")).count().orderBy("year").collect())
    write_csv(reports / "ratings_per_year.csv", ["year", "count"], [(x[0], x[1]) for x in years])
    for name, df in [("ratings_per_user.csv", per_user), ("ratings_per_movie.csv", per_item)]:
        rows = df.groupBy("n_ratings").count().orderBy("n_ratings").collect()
        write_csv(reports / name, ["n_ratings", "count"], [(x[0], x[1]) for x in rows])

    (reports / "data_quality.json").write_text(json.dumps(q, indent=2))
    (reports / "data_profile.json").write_text(json.dumps(p, indent=2))
    write_markdown(reports / "data_profile.md", q, p, dist)
    print(f"ingest done in {time.time() - t0:.0f}s: {n_r:,} ratings, {n_u:,} users, {n_i:,} movies")
    print(f"dropped {q['rows_dropped_total']:,} rows; details in {reports / 'data_quality.json'}")


def write_markdown(path: Path, q, p, dist):
    checks = [
        ("Missing userId / movieId / rating / timestamp",
         f"{q['missing_userId']} / {q['missing_movieId']} / {q['missing_rating']} / {q['missing_ts']}", "row dropped"),
        ("Unparseable userId / movieId / rating / timestamp",
         f"{q['unparseable_userId']} / {q['unparseable_movieId']} / {q['unparseable_rating']} / {q['unparseable_ts']}",
         "row dropped"),
        ("Rating outside 0.5-5.0 or not a half step", q["invalid_rating_value"], "row dropped"),
        ("Timestamp outside the collection period", q["timestamp_out_of_range"], "row dropped"),
        ("Rating for a movieId missing from movies.csv", q["ratings_orphan_movieId"], "row dropped"),
        ("Duplicate user-movie pairs (extra rows)", f"{q['duplicate_pairs']} ({q['duplicate_extra_rows']})",
         "latest rating kept"),
        ("... of which with different ratings", q["duplicate_pairs_with_different_ratings"], "latest rating kept"),
        ("Duplicate movieId in movies.csv", q["movies_duplicate_movieId"], "one row kept"),
        ("Movies without any rating", q["movies_without_ratings"], "no dense id, not recommendable"),
        ("Titles without a (year)", q["movies_title_without_year"], "kept (titles are not used by the methods)"),
        ("Movies with '(no genres listed)'", q["movies_without_genres"], "kept (genres are not used by the methods)"),
    ]
    stats = [
        ("Ratings (raw / clean)", f"{q['rows_raw']:,} / {p['ratings']:,}"),
        ("Users", f"{p['users']:,}"),
        ("Movies with at least one rating (in movies.csv)", f"{p['movies_with_ratings']:,} ({p['movies_in_movies_csv']:,})"),
        ("Density of the user-movie matrix", f"{p['density_pct']}%"),
        (f"Likes (rating >= {LIKE_THRESHOLD})", f"{p['likes']:,} ({p['like_share_pct']}%)"),
        ("Period", f"{p['first_rating_utc']} to {p['last_rating_utc']} UTC"),
        ("Ratings per user: min / median / mean / p99 / max",
         "{min} / {p50} / {mean} / {p99} / {max}".format(**p["ratings_per_user"])),
        ("Likes per user: min / median / mean / p99 / max",
         "{min} / {p50} / {mean} / {p99} / {max}".format(**p["likes_per_user"])),
        ("Ratings per movie: min / median / mean / p99 / max",
         "{min} / {p50} / {mean} / {p99} / {max}".format(**p["ratings_per_movie"])),
        ("Users with fewer than 20 ratings", p["users_with_fewer_than_20_ratings"]),
        ("Users without any like", p["users_without_any_like"]),
        ("Movies without any like / with only 1 rating", f"{p['movies_without_any_like']} / {p['movies_with_1_rating']}"),
        ("Share of ratings on the top 1% of movies", f"{p['share_of_ratings_on_top_1pct_movies_pct']}%"),
        ("Ratings that share a timestamp with another rating of the same user",
         f"{p['ratings_sharing_a_timestamp_with_same_user_pct']}%"),
    ]
    text = "\n\n".join([
        "# Data profile (generated by src/ingest.py)",
        "## Quality checks", md_table(["Check", "Rows found", "Action"], checks),
        "## Dataset after cleaning", md_table(["Statistic", "Value"], stats),
        "## Rating distribution", md_table(["Rating", "Count"], [(x[0], f"{x[1]:,}") for x in dist]),
    ])
    path.write_text(text + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW, help="folder with ratings.csv and movies.csv")
    ap.add_argument("--out", type=Path, default=PROCESSED)
    ap.add_argument("--reports", type=Path, default=REPORTS)
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()
    spark = get_spark("ingest", driver_memory=args.driver_memory)
    try:
        run(spark, args.raw, args.out, args.reports)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
