# Contributing

Working notes for everyone who changes or runs code in this repository: the group rules, the commands and the architecture.

ENSF 612 Group 9 project: scalable top-k movie recommendation on MovieLens 20M with PySpark (local mode).
This repo is the shared foundation (ingest, splits, evaluation, popularity baseline). Other methods
(exact user-kNN, MinHash-LSH, Monte Carlo PPR, ALS) plug in as one file each under `src/methods/`.
README.md holds the full list of decisions; it is written to be copied into the report, so keep it in
sync when behaviour changes.

## Group workflow (read first)

Four members finish the project in parallel, each on their own machine and branch; `tasks/README.md`
explains the whole flow. Each member (knn, lsh, ppr, als) works from `tasks/<method>.md`: find the first
unticked step and carry on from there, ticking steps as they are verified. Rules that always hold:

- **Shared code is frozen**: `src/`, `tests/`, `tools/`, `requirements.txt`. A member commits only the files
  listed for them in `tools/member.py` (`OWN`): `results/<method>/`, `report/<method>.md`, `tasks/<method>.md`
  (+ `report/data.md` for ppr, `report/evaluation.md` for als). Only the integrator (`tasks/integrate.md`) pushes to `main`.
- **Test stage once**: parameters are fixed in `tools/chosen.json` (chosen on validation). Run tests only with
  `python tools/member.py test <method>`; never re-run a split that has a `metrics.json`, never retune after
  seeing a test number.
- `python tools/member.py check|pilot|test|push <method>`: machine + data identity check, reproduction of the
  group's pilot validation metrics, final test runs, commit + push of branch `member/<method>`.
- `results/summary.csv` and `logs/` are not in git. `python tools/rebuild_summary.py` rebuilds the summary from
  the `metrics.json` files (run it before `tools/summarize.py`); `python tools/verify_results.py` checks that all
  test results are comparable (chosen params, same evaluated users as the baseline).
- Timings from different laptops are not comparable: all scalability runs (`python tools/scale.py`, steps in
  `tasks/scale.md`) are made by one member on one machine and pushed as branch `member/scale`.

## Commands

Run everything from the project root (scripts are run as modules: `python -m src.<name>`).

```bash
pip install -r requirements.txt               # pyspark==4.0.1, numpy, pytest
python -m pytest -q                           # 17 tests (pipeline + methods), ~3.5 min, real local Spark
python -m pytest -q tests/test_pipeline.py::test_metrics_by_hand   # single test
.\tools\run.ps1 <logname> -m src.run --method knn --split user_time_10pct --stage val   # sets Java/Hadoop, logs to logs\

python -m src.ingest                          # data/raw/ml-20m -> data/processed + reports/
python -m src.split                           # data/processed -> data/splits/{user_time,global_time,*_10pct}
python -m src.run --method popularity --split user_time_10pct --stage val
python -m src.run --method <name> --split user_time_10pct --stage val --param n_hashes=128 --tag n128
```

`src.run` flags: `--cores 1|2|4|*` (scalability runs), `--driver-memory 5g` (8 GB laptops; default 8g),
`--n-boot 0` (skip bootstrap for fast iteration), `--k`.

Synthetic data for quick end-to-end checks (overwrites `data/processed`; re-run ingest + split on real data afterwards):

```bash
python tools/make_synthetic.py --out data/raw/synthetic --dirty
python -m src.ingest --raw data/raw/synthetic
```

## Environment gotchas (Windows)

- Use Python 3.11. PySpark 4.0.x Python workers crash on Windows with 3.12+ (SPARK-53759).
- Native Windows needs `winutils.exe` + `hadoop.dll` in `%HADOOP_HOME%\bin` to write Parquet, and JDK 17/21 with `JAVA_HOME`.
- `common.use_this_python()` forces `PYSPARK_PYTHON` to the current interpreter; any new SparkSession
  (including in tests) must call it before creation, otherwise Windows picks a different python.
- If a shell reports `JAVA_GATEWAY_EXITED` / "Java not found", set `JAVA_HOME` (JDK 17), `HADOOP_HOME` (`C:\hadoop`) and
  prepend both `\bin` dirs to `PATH` in that command; `tools\run.ps1` and `tools\member.py` do this themselves.
  Run scripts with `.\.venv\Scripts\python.exe` and log to `logs\<step>.log`.
- Progress is tracked in `CHECKLIST.md` (project) and `tasks/<method>.md` (each member's steps).
- `ENSF612_ROOT` env var overrides the project root used for all data/results paths.

## Architecture

Pipeline: `ingest.py` → `split.py` → `run.py` (calls `methods/<name>.recommend`, then `evaluate.py`).

- **`src/common.py`** is the single source of project rules: `LIKE_THRESHOLD=4.0`, `TEST_PCT=20`, `VAL_PCT=10`,
  `K=10`, `SEED=612`, timestamp sanity window, all paths, `get_spark()` (shared Spark config: UTC, AQE,
  64 shuffle partitions), `user_sample()` and `load_stage()`. Changing a constant changes it for every
  method and the whole group; don't change one casually.
- **Data schema** (every Parquet table): `user`, `item` (dense ints from 0), `rating`, `ts` (Unix seconds),
  `like` (bool). `data/processed/{users,movies}` map dense ids back to `userId`/`movieId`/titles.
- **Stages** (`load_stage`): `val` = train → score on val (tune freely); `test` = train+val → score on test
  (run once per method with params chosen on val).
- **Splits**: `user_time` = per-user last ceil(20%) test, ceil(10%) before as val, ordered by (ts, item) so ties
  are deterministic. `global_time` = two global timestamp cutoffs. `*_10pct` pilots keep 10% of users.
- **`user_sample`** hashes the user id (xxhash64 with SEED), not `rand()`, so samples are nested
  (10% ⊂ 25% ⊂ 50%) and independent of partitioning/core count. Use it for all sampling.
- **Method contract** (`src/methods/_template.py`): `recommend(spark, train, k=10, **params) -> DataFrame(user, item, rank)`
  with rank 1..k, no duplicate items per user, never an item already rated in `train`, and never read
  val/test files inside a method. `--param key=value` values are parsed with `ast.literal_eval`.
- **Methods**: `popularity`, `knn` (exact user-kNN), `lsh` (MinHash banding + exact verification, same scoring
  as knn), `ppr` (Monte Carlo PPR on the like graph), `als` (MLlib implicit ALS + exact numpy scoring).
  All but popularity use `_shared.py`: `build_index` collects train into numpy CSR arrays (user→rated,
  user→liked, item→likers), broadcasts them, and scores fixed 256-user batches in `mapPartitions`.
  `top_k` never returns a seen item and fills short lists with popular items. Determinism rule: any top-n
  selection must go through `best_idx` (ties → lower id). Plain `argpartition` picks ties by input
  order, which changes with Spark partitioning and breaks core-count independence.
- **Evaluation** (`evaluate.py`): users with ≥1 liked target item and ≥1 train rating are evaluated; an evaluated
  user with no list scores 0. Precision/Recall/NDCG@k (binary relevance), coverage, 1000-resample user bootstrap
  95% CIs. `check_recs` rejects duplicate items/ranks and out-of-range ranks; warns on already-seen items.
- **Outputs**: `results/<method>/<split>/<stage>[__<tag>]/{recs,per_user,metrics.json}`; one row appended to
  `results/summary.csv` per run. `per_user` is kept for paired bootstrap comparisons between methods.
- **Cleaning rules** in `ingest.py` are each counted in `reports/data_quality.json` (even when 0). Users/movies
  are never dropped for being outliers; movies with no ratings get no dense id.

## Data

`data/raw/ml-20m/` (ratings.csv, movies.csv) is downloaded by each member from GroupLens and must not be committed.
