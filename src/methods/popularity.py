"""Step 5 - popularity baseline: the floor every other method must beat.

Each user gets the k most-liked training movies they have not rated yet. Popularity is the
number of likes in the training data; ties go to the movie with more ratings, then the lower id.
Movies with no like stay at the end of the order, so a user who has rated almost everything
still gets k items.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType


def recommend(spark: SparkSession, train: DataFrame, k: int = 10) -> DataFrame:
    order = (train.groupBy("item")
             .agg(F.sum(F.col("like").cast("int")).alias("likes"), F.count("*").alias("n"))
             .orderBy(F.desc("likes"), F.desc("n"), F.asc("item")))
    ranked = [r["item"] for r in order.select("item").collect()]  # ~27k ints: fine on the driver
    ranked_b = spark.sparkContext.broadcast(ranked)

    def top_unseen(seen):
        seen = set(seen)
        out = []
        for item in ranked_b.value:
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out

    top_unseen_udf = F.udf(top_unseen, ArrayType(IntegerType()))
    seen = train.groupBy("user").agg(F.collect_set("item").alias("seen"))
    return (seen.select("user", F.posexplode(top_unseen_udf("seen")).alias("pos", "item"))
            .select("user", "item", (F.col("pos") + 1).alias("rank")))
