"""Shared settings, paths and helpers. Every script imports from here, so a rule is defined once.

Changing a constant here changes it for every method; agree in the group before you do.
"""
import os
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------- project rules
LIKE_THRESHOLD = 4.0   # a rating >= 4.0 counts as a like (proposal, Section 3)
TEST_PCT = 20          # latest 20% of each user's ratings -> test (integers avoid float rounding)
VAL_PCT = 10           # the 10% before them -> validation
K = 10                 # length of every top-k list
SEED = 612             # one seed for all sampling
PILOT_FRAC = 0.10      # pilot sample: 10% of users

# Sanity window for timestamps (ml-20m README: Jan 09, 1995 to Mar 31, 2015).
TS_MIN = 789_609_600    # 1995-01-09 00:00:00 UTC
TS_MAX = 1_427_846_399  # 2015-03-31 23:59:59 UTC

# ---------------------------------------------------------------- paths
ROOT = Path(os.environ.get("ENSF612_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
RAW = DATA / "raw" / "ml-20m"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"


def use_this_python():
    """Make Spark's Python workers use the same interpreter as the driver (this venv).

    Spark starts workers with "python3" by default. A Windows venv has no python3.exe, so Windows
    finds another installation (e.g. a Microsoft Store Python) and Spark stops with
    PYTHON_VERSION_MISMATCH. Overrides any PYSPARK_PYTHON set elsewhere for the same reason.
    Must run before the first SparkSession is created.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    if sys.platform == "win32" and sys.version_info >= (3, 12):
        print("WARNING: PySpark 4.0.x workers crash on Windows with Python 3.12+ (SPARK-53759): "
              "'Python worker exited unexpectedly (crashed)'. Create the venv with Python 3.11, "
              "see README section 1.", file=sys.stderr)


def get_spark(app_name: str = "ensf612", cores: str = "*", driver_memory: str = "8g",
              shuffle_partitions: int = 64) -> SparkSession:
    """Local Spark session with the same settings for everyone.

    cores: "*" for all cores, or "1", "2", "4" for the scalability runs.
    driver_memory: in local mode the driver is also the executor, so this is Spark's whole heap.
    It only takes effect if no session is running yet in this Python process.
    """
    use_this_python()
    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def user_sample(df: DataFrame, frac: float, seed: int = SEED, col: str = "user") -> DataFrame:
    """Keep a deterministic fraction of users.

    Uses a hash of the user id, not rand(), so the result does not depend on partitioning or
    on the number of cores, and samples are nested: the 10% sample is inside the 25% sample,
    which is inside the 50% sample. The scalability runs (RQ3) should use this same function.
    """
    if frac >= 1.0:
        return df
    bucket = F.pmod(F.xxhash64(F.col(col), F.lit(seed)), F.lit(1_000_000))
    return df.where(bucket < F.lit(int(round(frac * 1_000_000))))


def split_dir(split: str) -> Path:
    return SPLITS / split


def load_stage(spark: SparkSession, split: str, stage: str):
    """Return (train, target) for one stage of an experiment.

    stage="val":  train = train part,        target = validation part (tune here)
    stage="test": train = train + validation, target = test part       (score once, at the end)
    """
    d = split_dir(split)
    train = spark.read.parquet(str(d / "train"))
    if stage == "val":
        return train, spark.read.parquet(str(d / "val"))
    if stage == "test":
        return train.unionByName(spark.read.parquet(str(d / "val"))), spark.read.parquet(str(d / "test"))
    raise ValueError("stage must be 'val' or 'test'")
