"""Template for a new method. Copy this file to src/methods/<name>.py and fill in recommend().

The contract every method follows (src/run.py relies on it):

    recommend(spark, train, k=10, **params) -> DataFrame

    train   DataFrame with columns
                user   int    dense user id (0..N-1)
                item   int    dense movie id (0..M-1)
                rating double 0.5..5.0
                ts     long   Unix seconds
                like   bool   rating >= 4.0
            For tuning it is the train part; for the final score it is train + validation.
            Never read the validation or test files inside a method.

    return  DataFrame with columns user (int), item (int), rank (int, 1..k)
            * one row per (user, rank), at most k rows per user, no repeated item for a user;
            * a list for every user in `train` if you can (missing users score 0);
            * never recommend an item the user already rated in `train`.

    params  hyper-parameters, passed from the command line:
                python -m src.run --method <name> --split user_time_10pct --stage val --param n_hashes=128
            Values are parsed as Python literals (128 -> int, 0.15 -> float, 'abc' -> str).

Then run it like the popularity baseline; results land in results/<name>/<split>/<stage>/.
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def recommend(spark: SparkSession, train: DataFrame, k: int = 10, **params) -> DataFrame:
    raise NotImplementedError("copy this template and implement your method")
