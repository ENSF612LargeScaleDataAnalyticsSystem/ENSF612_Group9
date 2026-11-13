# ENSF 612 Group 9: Scalable movie recommendation on MovieLens 20M

Shared foundation for the project: data checks and cleaning, the two time splits, the common
evaluation and the popularity baseline. The four other methods (exact user-kNN, MinHash-LSH,
Monte Carlo PPR, ALS) plug into the same pipeline through one function, described in Section 6.

## 1. Setup

You need Python 3.10 or newer (exactly 3.11 on Windows), Java 17 or 21, and about 10 GB of free disk space.
Group members: follow your task file instead (Section 8); it walks through all of this.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                # 17 tests, about 4 minutes; checks that Spark runs on your machine
```

**Windows: use Python 3.11.** With Python 3.12 or newer, every PySpark 4.0.x step that runs
Python code crashes on Windows with `Python worker exited unexpectedly (crashed)` (Spark bug
SPARK-53759). Create the venv with 3.11: `py -3.11 -m venv .venv`. Linux, macOS and WSL2 are
not affected.

**Windows: Hadoop files.** Spark on Windows cannot write Parquet without two extra Hadoop files.
- Easiest option: use WSL2 (Ubuntu) and follow the Linux steps above.
- Native Windows option: install JDK 17 and set `JAVA_HOME`. Download `winutils.exe` and
  `hadoop.dll` for Hadoop 3.x (PySpark 4.0.1 ships Hadoop 3.4.1) and put them in `C:\hadoop\bin`.
  Then set `HADOOP_HOME=C:\hadoop` and add `%HADOOP_HOME%\bin` to `PATH`. Without them you get
  `HADOOP_HOME and hadoop.home.dir are unset` or `UnsatisfiedLinkError` on the first write.
- Keep the project outside OneDrive and other synced folders: sync locks the files Spark writes.

**Memory.** The scripts start Spark with an 8 GB heap. On an 8 GB laptop add `--driver-memory 5g`.

## 2. Data

Download `ml-20m.zip` from <https://grouplens.org/datasets/movielens/20m/> and unzip it so that
`data/raw/ml-20m/ratings.csv` and `data/raw/ml-20m/movies.csv` exist. The GroupLens README asks
each user to download the data from them, so every member downloads it and nobody commits it.

## 3. Run the pipeline

Run from the project root, in this order:

```bash
python -m src.ingest                 # step 2: checks, cleaning, dense ids, data profile
python -m src.split                  # step 3: user_time and global_time splits + 10% pilots
python -m src.run --method popularity --split user_time_10pct --stage val    # quick check
python -m src.run --method popularity --split user_time --stage test
python -m src.run --method popularity --split global_time --stage test
```

To try everything in about two minutes without the real data:

```bash
python tools/make_synthetic.py --out data/raw/synthetic --dirty
python -m src.ingest --raw data/raw/synthetic
```

The synthetic data overwrites `data/processed`, so run `python -m src.ingest` and
`python -m src.split` again on the real data afterwards.

## 4. Layout and outputs

```
src/common.py          constants (like threshold, split sizes, k, seed), paths, Spark session, user_sample
src/ingest.py          raw CSV -> data/processed/{ratings,users,movies}
src/split.py           data/processed/ratings -> data/splits/<split>/{train,val,test}
src/evaluate.py        ground truth, Precision/Recall/NDCG@k, coverage, bootstrap intervals
src/run.py             runs any method on a split and stage, then evaluates it
src/methods/           one file per method: popularity, knn, lsh, ppr, als (+ _template.py);
                       _shared.py holds the numpy index, batching and top-k code the last four share
tools/run.ps1          Windows helper: sets JAVA_HOME/HADOOP_HOME and logs a run to logs\<name>.log
tools/make_synthetic.py  fake data in the ml-20m format, for testing
tools/member.py        group workflow: check / pilot / test / push for one member (Section 8)
tools/rebuild_summary.py, verify_results.py, summarize.py, compare.py   tables and checks for the report
tasks/                 one step-by-step task file per member
report/                report sections written by the members
tests/                 hand-checked unit tests
reports/               generated tables for the report: data_profile.md, data_quality.json,
                       split_stats.md, rating_distribution.csv, ratings_per_year.csv, ...
results/<method>/<split>/<stage>/   recs, per_user metrics, metrics.json
results/summary.csv    one line per run; not in git, rebuilt by tools/rebuild_summary.py
```

Every Parquet table uses the same columns: `user` and `item` (dense ids starting at 0),
`rating`, `ts` (Unix seconds) and `like` (rating >= 4.0). `data/processed/users` and
`data/processed/movies` map the dense ids back to `userId`, `movieId` and titles.

## 5. Decisions, all of them in code (copy them into the report)

**Cleaning** (`src/ingest.py`). Every rule is counted in `reports/data_quality.json`, even when the
count is 0:
1. Drop rating rows with a missing or unparseable field.
2. Drop ratings outside 0.5–5.0 or not on a half-star step.
3. Drop ratings with a timestamp outside the collection period (9 Jan 1995 – 31 Mar 2015).
4. Drop ratings whose `movieId` is not in `movies.csv`.
5. When the same user–movie pair appears twice, keep the latest rating.

Movies without any rating get no dense id, because no method can recommend them. Users and movies
are never removed for being unusual, for example for having very many ratings.

**Likes.** A rating of 4.0 or more is a like.

**user_time split.** Each user's ratings are sorted by (timestamp, item). The last ceil(20% of n)
ratings are test and the ceil(10% of n) before them are validation. The item id breaks ties,
because many ratings share a timestamp. Every user has at least 20 ratings, so every user keeps at
least 13 for training (21 ratings give 5 to test and 3 to validation).

**global_time split.** Two cut-off timestamps shared by all users. The latest 20% of all ratings
are test and the 10% before them are validation. The dates are in `reports/split_stats.md`.

**Pilot samples.** `user_time_10pct` and `global_time_10pct` keep 10% of users with all their
ratings. `common.user_sample` picks users by a hash of the id, so samples are nested (10% ⊂ 25% ⊂ 50%)
and do not depend on the number of cores.

**Stages.** `--stage val` trains on train and scores on validation; tune here as often as you need.
`--stage test` trains on train + validation and scores on test. Run it once per method, with the
parameters chosen on validation.

**Evaluation** (`src/evaluate.py`):
- A user is evaluated if they have at least one liked item in the target part and at least one
  training rating.
- An evaluated user without a list scores 0. A method cannot gain by skipping hard users.
- Precision@10 = hits/10, Recall@10 = hits/liked target items, NDCG@10 with binary relevance and
  1/log2(rank+1) discounts. All are averaged over evaluated users.
- Coverage = distinct recommended items / items in the training part.
- 95% intervals come from 1,000 bootstrap resamples of users. Per-user values are saved in
  `results/.../per_user`, so methods can later be compared user by user (paired bootstrap).
- The evaluation rejects duplicate items, duplicate ranks and ranks outside 1..k. It warns when a
  method recommends a movie the user already rated.

## 6. Adding your method

1. Copy `src/methods/_template.py` to `src/methods/<name>.py` (for example `lsh.py`).
2. Implement `recommend(spark, train, k=10, **params)`. It takes the training DataFrame and returns
   `(user, item, rank)` with rank 1..k. Never recommend an item the user rated in `train`, and
   never read the validation or test files inside the method.
3. Develop on the pilot: `python -m src.run --method <name> --split user_time_10pct --stage val --param n_hashes=128 --tag n128`
4. Tune on `--stage val`, then run `--stage test` once on `user_time` and once on `global_time`.

Results from all methods then go into `results/summary.csv` in the same format.

## 7. Scalability runs (RQ3)

- Use `common.user_sample(df, frac)` for the 10/25/50/100% user samples.
- Use `--cores 1`, `--cores 2` and `--cores 4` with the same `--driver-memory` in every run.
- `metrics.json` already records wall-clock time, Spark version, CPU count and platform.
- Not built yet: shuffle bytes and peak memory per run. They can be read from the Spark REST API
  while the application is running (`/api/v1/applications/<id>/stages` and `/executors`).

## 8. Group workflow

Tuning is done and the chosen parameters are fixed in `tools/chosen.json`. The remaining work (final test
runs, scalability runs, report) is split between the four members: see `tasks/README.md` for who does
what, the steps each member follows, and how the branches are merged.

```bash
python tools/member.py check <method>    # machine ready, data and splits identical to the group's
python tools/member.py pilot <method>    # reproduces the group's validation numbers on the 10% pilot
python tools/member.py test  <method>    # final test runs, once per split, chosen parameters only
python tools/member.py push  <method>    # commits the member's own files, pushes branch member/<method>
python tools/scale.py                    # the 30 scalability runs (one member, one machine; tasks/scale.md)
```

## Status

Run on the real ml-20m files: ingest (20,000,263 ratings, 0 rows dropped), both splits, the popularity
baseline on both test splits, and validation tuning of kNN, LSH, PPR and ALS on the 10% pilot
(`reports/results.md`). Final test runs: kNN `user_time` is done; the rest are in the members' task files.
`CHECKLIST.md` has the details.
