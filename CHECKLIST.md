# Project checklist (ENSF 612 Group 9)

Tick items as they are done. Run commands from the project root. `.\tools\run.ps1 <log> <python args>`
sets JAVA_HOME/HADOOP_HOME and writes `logs\<log>.log`.

## 1. Setup
- [x] Python 3.11 venv, `pip install -r requirements.txt`
- [x] Java 17 + Hadoop `winutils.exe` / `hadoop.dll` working
- [x] `python -m pytest -q` passes: 17 tests (6 pipeline + 11 methods), ~3.5 min

## 2. Data and ingest
- [x] `data/raw/ml-20m/` downloaded
- [x] `python -m src.ingest` (342 s): 20,000,263 ratings, 138,493 users, 26,744 movies, 0 rows dropped
- [x] `reports/data_profile.md` matches the official ml-20m counts
- [ ] Copy the data profile and cleaning rules (README §5) into the report

## 3. Splits
- [x] `python -m src.split` (418 s): `user_time`, `global_time`, `user_time_10pct`, `global_time_10pct`
- [x] `reports/split_stats.md` generated
- [ ] Note for the report: at the test stage `global_time` evaluates only 5,542 users (26,027 test users have
      no training history), and 15% of its test likes are on movies unseen in training.

## 4. Popularity baseline
- [x] Pilot `user_time_10pct` val: P@10 0.0382, R@10 0.0726, NDCG@10 0.0639 (28 s)
- [x] `user_time` test: P@10 0.0619, R@10 0.0619, NDCG@10 0.0806, coverage 0.013 (38 s + 75 s eval)
- [x] `global_time` test: P@10 0.1262, R@10 0.0337, NDCG@10 0.1438, coverage 0.027 (36 s + 54 s eval)

## 5. Methods (`src/methods/`; shared helpers in `_shared.py`; tests in `tests/test_methods.py`)
Each method: implement → pilot run → tune on `user_time_10pct --stage val` → `--stage test` once on `user_time`
and once on `global_time`, with the chosen params.

| Method | File | Implemented + tested | Pilot run | Tuned (val) | user_time test | global_time test |
|---|---|---|---|---|---|---|
| Exact user-kNN | `knn.py` | [x] | [x] | [x] cosine, 100 neighbours (val NDCG 0.1081) | [x] NDCG 0.1367 | [ ] crashed once before scoring; rerun |
| MinHash-LSH | `lsh.py` | [x] | [x] | [x] 256 hashes, 128 bands, cosine, 100 neighbours (0.1068) | [ ] | [ ] |
| Monte Carlo PPR | `ppr.py` | [x] | [x] | [x] alpha 0.3, 2500 walkers, beta 0.3, min_visits 10 (0.0800) | [ ] | [ ] |
| ALS (MLlib) | `als.py` | [x] | [x] | [x] rank 128, alpha 1, reg 0.1 (0.1170) | [ ] | [ ] |

Tuning is finished; the chosen parameters are in `tools/chosen.json`. The remaining test runs, the
scalability runs and the report are split between the four members: see `tasks/README.md`. Each member
ticks their own `tasks/<method>.md`; the integrator ticks this file after merging (tasks/knn.md, step D8).

Pilot (`user_time_10pct` val, default params, 12,905 users):

| Method | P@10 | R@10 | NDCG@10 | Coverage | Recommend time |
|---|---|---|---|---|---|
| popularity | 0.0382 | 0.0726 | 0.0639 | 0.010 | 28 s |
| knn (n=50, jaccard) | 0.0624 | 0.1275 | 0.1057 | 0.079 | 57 s |
| lsh (128 hashes, 32 bands) | 0.0463 | 0.0994 | 0.0801 | 0.154 | 112 s |
| ppr (alpha 0.3, beta 0) | 0.0356 | 0.0731 | 0.0608 | 0.050 | 67 s |
| als (rank 64, implicit likes) | 0.0610 | 0.1275 | 0.1019 | 0.119 | 116 s |

Pilot findings: LSH with r = 4 rows per band has a Jaccard threshold of ~0.42, too high for this data
(24 candidates per user, 4,027 users with none), so the grid moved to r = 1–2. PPR without a popularity
penalty is below popularity, so the grid adds `beta`.

**Unattended chain** (`.\tools\after_tuning.ps1`, progress in `logs\after_tuning.log`), started 13:56:
waits for tuning → `tools\pick.py` writes `tools\chosen.json` → LSH pilot check with the chosen settings →
`tools\lsh_recall.py` → `tools\final.ps1` (test runs) → `tools\scale.ps1` (RQ3) → `tools\summarize.py` (`reports\results.md`).

PPR finding: more walkers helps a lot (100 → 250 walkers: NDCG 0.0608 → 0.0689, above popularity) at almost no
extra time. The popularity penalty beta collapses with few walkers (beta 0.5 → NDCG 0.0002): rare movies
visited once get the biggest boost. Added `min_visits` to ppr.py and a second round `tune.ps1 ppr2`
(1000/2500 walkers, beta with min_visits); the chain (restarted 14:33) runs it before picking.
LSH pick is limited to ≥ 2 rows per band (`tools\pick.py`): r = 1 is slower than exact kNN and would not finish
on the full data.

ALS finding: the implicit confidence alpha matters most: alpha 1 → NDCG 0.1127 (best of all methods), 10 → 0.1019,
40 → 0.0870; raw ratings as confidence → 0.0659. Rank 128 and reg 0.1 help slightly. Second round `tune.ps1 als2`
(alpha 0.3 / 3, rank 128 with alpha 1) added to the chain (restarted ~14:45).

Main tuning grid (26 configs) finished at ~14:44, all exit 0.

Tuning so far: kNN best = cosine, 100 neighbours (NDCG 0.1081). LSH r=2 bands: (128,64) 0.1027, (256,128) 0.1049,
against exact Jaccard kNN at 0.1065.

Tuning: `.\tools\tune.ps1 knn lsh ppr als` (grids inside the script; log `logs\tune_all.log`).
Pick each method's best NDCG@10 in `results/summary.csv` (rows with stage `val`, split `user_time_10pct`).
LSH neighbour quality: `python tools/lsh_recall.py --n-hashes 128 --bands 64`.

## 6. Scalability (RQ3)
- [x] Results do not depend on the core count (fixed user batches, ties broken by id, seeded per batch, fixed ALS blocks)
- [x] `src.run --sample 0.25` option (uses `common.user_sample` on train and target; adds `s25` to the tag)
- [ ] 10 / 25 / 50 / 100% user samples for each method
- [ ] `--cores 1`, `2`, `4` with the same `--driver-memory` for each method
- [ ] LSH vs exact kNN: neighbour recall and time, against (`n_hashes`, `bands`)
- [ ] Not built yet: shuffle bytes and peak memory per run (Spark REST API `/api/v1/applications/<id>/stages`, `/executors`)

## 7. Report
- [ ] Results tables from `results/summary.csv` with 95% bootstrap CIs
- [ ] Paired bootstrap comparisons: `python tools/compare.py results/<A>/user_time/test results/<B>/user_time/test`
- [ ] Scalability plots
