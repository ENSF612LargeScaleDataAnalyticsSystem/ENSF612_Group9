# Task: ALS (Spark MLlib, implicit feedback) (`als`)

You are Claude Code working for the group member who owns **als**. Do the steps in order. After a step is
done **and verified**, change its `- [ ]` to `- [x]` in this file and go on to the next one. When the
member says "continue", find the first unticked step and carry on from there. Do not stop between
steps unless a step says **STOP**.

Run every command from the project root with the project's venv Python: `.venv\Scripts\python.exe`
on Windows, `.venv/bin/python` on macOS/Linux (written as `python` below).

## Rules (they protect the comparison between the four methods)

1. **Never edit** `src/`, `tests/`, `tools/`, `requirements.txt`, `reports/`, `README.md`, `CLAUDE.md`,
   `CHECKLIST.md` or another member's files. Your files are: `results/als/`, `report/als.md`, `report/evaluation.md`, `tasks/als.md`.
2. **The test stage runs once.** Parameters are fixed in `tools/chosen.json` (rank=128, alpha=1.0, reg=0.1); they were chosen
   on validation. Never change them, never pass `--param` yourself, never re-run a test that produced a
   `metrics.json`, whatever the numbers look like. Always use `tools/member.py test`, which enforces this.
3. If a run crashes **before** writing `metrics.json`, nothing was scored and running it again is allowed.
4. If something in the shared code looks wrong, **STOP** and tell the member to raise it with the group.
   Do not fix it locally.
5. Long commands (ingest, split, pilot, test) take minutes to hours: start them in the background and
   wait for them to finish. Do not start two Spark jobs at once.
6. Every number you write in a report file must be copied from a `metrics.json` or a log of a run on
   this machine or in the repository. No estimates, no rounding beyond 4 decimals.

## Steps

- [ ] **1. Get the repository and your branch.**
  If the project is not on this machine yet: `git clone https://github.com/ENSF612LargeScaleDataAnalyticsSystem/ENSF612_Group9.git` and open that folder. Keep it outside
  OneDrive or any synced folder. Then:
  ```
  git fetch origin
  git checkout member/als 2>nul || git checkout -b member/als origin/main
  ```
  (macOS/Linux: `2>/dev/null` instead of `2>nul`.) Verify: `git branch --show-current` prints `member/als`.

- [ ] **2. Python environment.**
  Windows: `py -3.11 -m venv .venv` (Python 3.11 exactly; 3.12+ crashes PySpark on Windows).
  macOS/Linux: `python3 -m venv .venv` with Python 3.10 or newer.
  Then `python -m pip install -r requirements.txt`.
  If Python 3.11 is missing on Windows, **STOP** and ask the member to install it
  (`winget install Python.Python.3.11`), then continue.

- [ ] **3. Java and, on Windows, the Hadoop files.**
  Needed: JDK 17 with `JAVA_HOME` set; on Windows also `winutils.exe` and `hadoop.dll` (Hadoop 3.x) in
  `C:\hadoop\bin` with `HADOOP_HOME=C:\hadoop`. README section 1 has the details. Check what is already
  there before changing anything. Installing software or setting user environment variables changes the
  member's machine: **STOP and ask the member first**, then do it (`winget install EclipseAdoptium.Temurin.17.JDK`,
  `setx JAVA_HOME "..."`, `setx HADOOP_HOME C:\hadoop`).

- [ ] **4. Data.**
  Download <https://files.grouplens.org/datasets/movielens/ml-20m.zip> (about 190 MB) and unzip it so
  that `data/raw/ml-20m/ratings.csv` and `data/raw/ml-20m/movies.csv` exist. Never commit `data/`.

- [ ] **5. Unit tests.** `python -m pytest -q` → 17 passed (about 4 minutes). If any fail, fix the
  environment (steps 2–3), not the code.

- [ ] **6. Ingest and split** (about 6 and 7 minutes on the group's machine; on 8 GB RAM add `--driver-memory 5g`):
  ```
  python -m src.ingest > logs/ingest.log 2>&1
  python -m src.split  > logs/split.log 2>&1
  ```
  (Create `logs/` first if it does not exist.) Both logs must end without a Python traceback.

- [ ] **7. Machine and data check.** `python tools/member.py check als` must print `ALL CHECKS PASSED`.
  It verifies the tools, the branch, that shared code is untouched, and that this machine's data profile
  and split statistics are identical to the group's. Fix each `FAIL` using the printed hint and run it
  again. If the data or split check keeps failing, **STOP**: this machine would test on different rows.

- [ ] **8. Reproduce the group's validation number.** `python tools/member.py pilot als` (a few minutes;
  on 8 GB RAM add `--driver-memory 5g`). It re-runs the chosen parameters on the 10% pilot, validation
  stage, and must print `PILOT REPRODUCED` (the group's NDCG@10 is 0.1170). If it prints `FAIL`,
  **STOP** and report the numbers to the member; do not go on to the test stage.

- [ ] **9. Final test runs.** Start in the background and wait (from half an hour to a few hours in total):
  ```
  python tools/member.py test als
  ```
  (on 8 GB RAM: `python tools/member.py test als --driver-memory 5g`). It runs `user_time` then
  `global_time`, skips a split that already has a result, and must end with `TEST RUNS COMPLETE`.
  If a split fails, read the end of `logs/final_als_<split>.log`, fix the environment cause (memory,
  disk space, another Spark job running) and run the same command again. If it fails twice, **STOP**.

- [ ] **10. Push the results now**, so they are safe: `python tools/member.py push als`.
  If the push asks for a GitHub login, **STOP** and ask the member to sign in, then run it again.

- [ ] **11. Compare with the popularity baseline** (paired bootstrap, a few minutes each):
  ```
  python tools/compare.py results/als/user_time/test   results/popularity/user_time/test   > logs/compare_als_pop_user_time.log 2>&1
  python tools/compare.py results/als/global_time/test results/popularity/global_time/test > logs/compare_als_pop_global_time.log 2>&1
  ```

- [ ] **12. Report section on the evaluation protocol (`report/evaluation.md`).**
  Write `report/evaluation.md` for the whole group. Sources: README section 5 (stages, evaluation),
`src/evaluate.py`, `src/run.py`, `tools/compare.py`. Do not run anything.

Outline:
1. **Stages**: validation (train → val, used for tuning on the 10% pilot) and test (train + val → test,
   run once per method with the parameters fixed in `tools/chosen.json`).
2. **Who is evaluated**: users with at least one liked item in the target part and at least one training
   rating; an evaluated user without a list scores 0.
3. **Metrics**: Precision@10, Recall@10, NDCG@10 (binary relevance), coverage; exact definitions.
4. **Uncertainty**: 95% intervals from 1,000 bootstrap resamples of users; paired bootstrap for
   comparing two methods on the same users.
5. **Why there is no leakage**: methods receive only the training DataFrame; time-ordered splits; tuning
   never touched the test part; every member's machine was checked to hold the identical split
   (`tools/member.py check`) and to reproduce the group's validation numbers (`tools/member.py pilot`).
6. **Reproducibility**: results do not depend on the number of cores (fixed user batches, ties broken by
   id, seeded per batch).

Then compare ALS with exact kNN on `user_time` (the kNN run is already in the repository):

```
python tools/compare.py results/als/user_time/test results/knn/user_time/test > logs/compare_als_knn.log 2>&1
```

- [ ] **13. Write `report/als.md`** (create the `report/` folder if needed). Plain, short sentences;
  about one page. Outline:

  ## 1. Method
  What ALS (Spark MLlib, implicit feedback) does and how it is implemented here on Spark. Read `src/methods/als.py` and
  `src/methods/_shared.py`; describe what the code does, not the textbook version.

  ## 2. Parameters and tuning
  A table of every validation run of this method on the pilot: read each
  `results/als/user_time_10pct/val__*/metrics.json` (tag, parameters, P@10, R@10, NDCG@10, coverage,
  recommend seconds), sorted by NDCG@10, with the chosen configuration marked. State that tuning used
  only the validation stage of the 10% pilot.

  ## 3. Test results
  One row per split (`user_time`, `global_time`) from `results/als/<split>/test/metrics.json`:
  P@10, R@10, NDCG@10 each with its 95% interval (`*_ci95`), coverage, evaluated users, recommend
  seconds. Add the popularity baseline rows from `results/popularity/<split>/test/metrics.json`.
  Under the table, state the machine the runs were made on (`env` in `metrics.json`: CPU count, platform,
  driver memory).

  ## 4. Comparison with the baseline
  The paired differences from step 11 with their 95% intervals, for both splits, and one or two
  sentences on what they mean. Report what the numbers say even if the method is worse than the baseline.

  ## 5. Findings from tuning and comparison with kNN
The confidence weight `alpha` mattered most (alpha 1 best; 10 and 40 worse; raw ratings as confidence much
worse); rank 128 and reg 0.1 helped slightly. Take the numbers from the `user_time_10pct` runs. Add the
paired ALS − kNN differences on `user_time` with their 95% intervals.

  ## 6. Limitations
  Two or three honest points (for example memory use, run time, what `global_time` can evaluate).

- [ ] **14. Final push.** `python tools/member.py push als -m "als: report section"`. It prints a link:
  give that link to the member and ask them to open it in a browser and press **Create pull request**
  (into `main`). Do not merge it; the integrator merges.

- [ ] **15. Done.** Tell the member, in a short message: the test numbers for both splits, where the
  report section is, the pull-request link, and anything that went wrong on the way.
