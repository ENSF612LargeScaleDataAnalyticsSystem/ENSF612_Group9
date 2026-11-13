# Task: MinHash-LSH (`lsh`)

These are the steps for the group member who owns **lsh**. Do the steps in order. After a step is
done **and verified**, change its `- [ ]` to `- [x]` in this file and go on to the next one. When the
member says "continue", find the first unticked step and carry on from there. Do not stop between
steps unless a step says **STOP**.

Run every command from the project root with the project's venv Python: `.venv\Scripts\python.exe`
on Windows, `.venv/bin/python` on macOS/Linux (written as `python` below).

## Rules (they protect the comparison between the four methods)

1. **Never edit** `src/`, `tests/`, `tools/`, `requirements.txt`, `reports/`, `README.md`,
   `CHECKLIST.md`, the other files in the project root or another member's files. Your files are: `results/lsh/`, `report/lsh.md`, `tasks/lsh.md`.
2. **The test stage runs once.** Parameters are fixed in `tools/chosen.json` (n_hashes=256, bands=128, similarity=cosine, n_neighbours=100); they were chosen
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

- [x] **1. Get the repository and your branch.**
  If the project is not on this machine yet: `git clone https://github.com/ENSF612LargeScaleDataAnalyticsSystem/ENSF612_Group9.git` and open that folder. Keep it outside
  OneDrive or any synced folder. Then:
  ```
  git fetch origin
  git checkout member/lsh 2>nul || git checkout -b member/lsh origin/main
  ```
  (macOS/Linux: `2>/dev/null` instead of `2>nul`.) Verify: `git branch --show-current` prints `member/lsh`.

- [x] **2. Python environment.**
  Windows: `py -3.11 -m venv .venv` (Python 3.11 exactly; 3.12+ crashes PySpark on Windows).
  macOS/Linux: `python3 -m venv .venv` with Python 3.10 or newer.
  Then `python -m pip install -r requirements.txt`.
  If Python 3.11 is missing on Windows, **STOP** and ask the member to install it
  (`winget install Python.Python.3.11`), then continue.

- [x] **3. Java and, on Windows, the Hadoop files.**
  Needed: JDK 17 with `JAVA_HOME` set; on Windows also `winutils.exe` and `hadoop.dll` (Hadoop 3.x) in
  `C:\hadoop\bin` with `HADOOP_HOME=C:\hadoop`. README section 1 has the details. Check what is already
  there before changing anything. Installing software or setting user environment variables changes the
  member's machine: **STOP and ask the member first**, then do it (`winget install EclipseAdoptium.Temurin.17.JDK`,
  `setx JAVA_HOME "..."`, `setx HADOOP_HOME C:\hadoop`).

- [x] **4. Data.**
  Download <https://files.grouplens.org/datasets/movielens/ml-20m.zip> (about 190 MB) and unzip it so
  that `data/raw/ml-20m/ratings.csv` and `data/raw/ml-20m/movies.csv` exist. Never commit `data/`.

- [x] **5. Unit tests.** `python -m pytest -q` → 17 passed (about 4 minutes). If any fail, fix the
  environment (steps 2–3), not the code.

- [x] **6. Ingest and split** (about 6 and 7 minutes on the group's machine; on 8 GB RAM add `--driver-memory 5g`):
  ```
  python -m src.ingest > logs/ingest.log 2>&1
  python -m src.split  > logs/split.log 2>&1
  ```
  (Create `logs/` first if it does not exist.) Both logs must end without a Python traceback.

- [x] **7. Machine and data check.** `python tools/member.py check lsh` must print `ALL CHECKS PASSED`.
  It verifies the tools, the branch, that shared code is untouched, and that this machine's data profile
  and split statistics are identical to the group's. Fix each `FAIL` using the printed hint and run it
  again. If the data or split check keeps failing, **STOP**: this machine would test on different rows.

- [x] **8. Reproduce the group's validation number.** `python tools/member.py pilot lsh` (a few minutes;
  on 8 GB RAM add `--driver-memory 5g`). It re-runs the chosen parameters on the 10% pilot, validation
  stage, and must print `PILOT REPRODUCED` (the group's NDCG@10 is 0.1068). If it prints `FAIL`,
  **STOP** and report the numbers to the member; do not go on to the test stage.

- [x] **9. Final test runs.** Start in the background and wait (from half an hour to a few hours in total):
  ```
  python tools/member.py test lsh
  ```
  (on 8 GB RAM: `python tools/member.py test lsh --driver-memory 5g`). It runs `user_time` then
  `global_time`, skips a split that already has a result, and must end with `TEST RUNS COMPLETE`.
  If a split fails, read the end of `logs/final_lsh_<split>.log`, fix the environment cause (memory,
  disk space, another Spark job running) and run the same command again. If it fails twice, **STOP**.

- [ ] **10. Push the results now**, so they are safe: `python tools/member.py push lsh`.
  If the push asks for a GitHub login, **STOP** and ask the member to sign in, then run it again.

- [ ] **11. Compare with the popularity baseline** (paired bootstrap, a few minutes each):
  ```
  python tools/compare.py results/lsh/user_time/test   results/popularity/user_time/test   > logs/compare_lsh_pop_user_time.log 2>&1
  python tools/compare.py results/lsh/global_time/test results/popularity/global_time/test > logs/compare_lsh_pop_global_time.log 2>&1
  ```

- [ ] **12. LSH against exact kNN (approximation quality).**
  Two measurements, both needed for the report.

1. Neighbour quality on the full data (uses the training part of the validation stage, never the test part).
   Run in the background; it can take 30+ minutes:

   ```
   python tools/lsh_recall.py --split user_time --n-hashes 256 --bands 128 --similarity cosine --n-neighbours 100 --users 2000 > logs/lsh_recall_full.log 2>&1
   ```

   Copy from the log: neighbour recall@100, similarity ratio, candidates per user, candidate-step time.
   The same measurement on the 10% pilot (group machine) gave: recall@100 0.8247, similarity ratio 0.9464,
   1,885.7 candidates per user, candidate step 78 s.

2. Recommendation quality against exact kNN, user by user. The kNN `user_time` test run is already in the
   repository:

   ```
   python tools/compare.py results/lsh/user_time/test results/knn/user_time/test > logs/compare_lsh_knn.log 2>&1
   ```

   Copy the three differences with their 95% intervals. (`global_time` kNN is produced by the kNN member;
   the integrator adds that comparison.)

- [ ] **13. Write `report/lsh.md`** (create the `report/` folder if needed). Plain, short sentences;
  about one page. Outline:

  ## 1. Method
  What MinHash-LSH does and how it is implemented here on Spark. Read `src/methods/lsh.py` and
  `src/methods/_shared.py`; describe what the code does, not the textbook version.

  ## 2. Parameters and tuning
  A table of every validation run of this method on the pilot: read each
  `results/lsh/user_time_10pct/val__*/metrics.json` (tag, parameters, P@10, R@10, NDCG@10, coverage,
  recommend seconds), sorted by NDCG@10, with the chosen configuration marked. State that tuning used
  only the validation stage of the 10% pilot.

  ## 3. Test results
  One row per split (`user_time`, `global_time`) from `results/lsh/<split>/test/metrics.json`:
  P@10, R@10, NDCG@10 each with its 95% interval (`*_ci95`), coverage, evaluated users, recommend
  seconds. Add the popularity baseline rows from `results/popularity/<split>/test/metrics.json`.
  Under the table, state the machine the runs were made on (`env` in `metrics.json`: CPU count, platform,
  driver memory).

  ## 4. Comparison with the baseline
  The paired differences from step 11 with their 95% intervals, for both splits, and one or two
  sentences on what they mean. Report what the numbers say even if the method is worse than the baseline.

  ## 5. Approximation quality
Neighbour recall and similarity ratio (pilot and full data), candidates per user, and the paired
LSH − kNN differences on `user_time`. Say in one sentence what is given up and what is gained: compare
`seconds_recommend` of LSH and kNN **only if both ran on the same machine** (check `env` in the two
`metrics.json` files); otherwise say the time comparison is in the scalability section.

  ## 6. Limitations
  Two or three honest points (for example memory use, run time, what `global_time` can evaluate).

- [ ] **14. Final push.** `python tools/member.py push lsh -m "lsh: report section"`. It prints a link:
  give that link to the member and ask them to open it in a browser and press **Create pull request**
  (into `main`). Do not merge it; the integrator merges.

- [ ] **15. Done.** Tell the member, in a short message: the test numbers for both splits, where the
  report section is, the pull-request link, and anything that went wrong on the way.
