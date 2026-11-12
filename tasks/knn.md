# Task: exact user-kNN (`knn`) and integration

These are the steps for the group member who owns **knn** and who is also the **integrator**: the only
person who pushes to `main`. The integrator's machine is the group's reference machine (tuning was done
there, and all scalability timings are made there so that they are comparable).

Do the steps in order. After a step is done **and verified**, change its `- [ ]` to `- [x]` in this file
and go on, so the work can always be picked up again at the first unticked step. Do not stop between
steps unless a step says **STOP**.

Run commands from the project root. Spark runs go through `.\tools\run.ps1 <logname> <python args>` or
`.\.venv\Scripts\python.exe tools\member.py ...` (both set Java/Hadoop).

## Rules

1. **The test stage runs once per method and split**, with the parameters in `tools/chosen.json`. Never
   re-run a test that produced a `metrics.json`. Use `tools/member.py test`, which enforces this.
2. After Part A, shared code (`src/`, `tests/`, `tools/`, `requirements.txt`) changes only for a real bug,
   on `main`, with a message to the group so that everybody pulls before testing.
3. Long commands run in the background, one Spark job at a time.
4. Every number in a report file is copied from a `metrics.json` or a log.
5. Pushing publishes to the group: before each `git push` to `main`, review what will be
   pushed (`git status`, `git log --oneline origin/main..HEAD`).

## Part A: publish the foundation (once)

- [x] **A1. Create the local repository.**
  ```
  git init -b main
  git add -A
  git status --short
  ```
  Verify the staged list has **no** `data/`, `.venv/`, `logs/`, `results/**/recs/` or `results/summary.csv`,
  and that the total is small (about 10 MB: `git count-objects -vH` after the commit).
  ```
  git commit -m "Shared foundation: pipeline, five methods, validation tuning, group workflow"
  ```

- [x] **A2. Push to GitHub.** Review what will be pushed (rule 5). Then:
  ```
  git remote get-url origin   # already set to the group repository
  git push -u origin main
  ```
  If GitHub asks for a login, sign in in the browser window that opens. If the push is
  rejected because the GitHub repository already has commits (for example a README created on the
  website), run `git pull origin main --allow-unrelated-histories --no-rebase`, keep our files in any
  conflict, and push again. Never force-push.

- [ ] **A3. Start the team.** Tell each teammate which task file is theirs (`tasks/lsh.md`, `tasks/ppr.md`,
  `tasks/als.md`) and check that all three have write access to the repository.

## Part B: kNN test runs and report section

- [ ] **B1. Branch.** `git checkout -b member/knn` (or `git checkout member/knn` if it exists).

- [ ] **B2. Check.** `.\.venv\Scripts\python.exe tools\member.py check knn` → `ALL CHECKS PASSED`.

- [ ] **B3. Test runs.** In the background:
  ```
  .\.venv\Scripts\python.exe tools\member.py test knn
  ```
  `user_time` is already done (12 Nov, NDCG@10 0.1367, recommend 1,634 s) and will be skipped.
  `global_time` was started on 12 Nov and crashed at 16:29 with "Python worker exited unexpectedly"
  before any result was written, so nothing was scored and this run is its first real one. If it
  crashes again, read `logs\final_knn_global_time.log`; close other heavy programs and try once more
  with `--driver-memory 6g` (this machine has 16 GB; the Python workers need the rest). If it fails
  twice, **STOP**.

- [ ] **B4. Compare with the baseline.**
  ```
  .\tools\run.ps1 compare_knn_pop_user_time   tools\compare.py results\knn\user_time\test   results\popularity\user_time\test
  .\tools\run.ps1 compare_knn_pop_global_time tools\compare.py results\knn\global_time\test results\popularity\global_time\test
  ```

- [ ] **B5. Write `report/knn.md`.** About one page, plain sentences:
  1. **Method**: what exact user-kNN does and how it is implemented (read `src/methods/knn.py` and `_shared.py`).
  2. **Parameters and tuning**: table of every `results/knn/user_time_10pct/val__*/metrics.json`, sorted by
     NDCG@10, chosen configuration marked (cosine, 100 neighbours); tuning used only the validation stage of the pilot.
  3. **Test results**: both splits with 95% intervals, coverage, evaluated users, recommend seconds, plus the
     popularity rows; state the machine (`env` in `metrics.json`).
  4. **Comparison with the baseline**: paired differences from B4, both splits.
  5. **Limitations**: quadratic cost in users, memory of the broadcast index, what `global_time` can evaluate.

- [ ] **B6. Push the branch.** `.\.venv\Scripts\python.exe tools\member.py push knn`

## Part C: scalability runs (RQ3), all methods, this machine only

- [ ] **C1. Run.** `git checkout main`, then in the background (several hours; do not use the machine for
  other heavy work, the timings are the result):
  ```
  .\tools\scale.ps1 *> logs\scale_all.log
  ```
  It runs every method on 10 / 25 / 50 / 100% of users with 4 cores, and on the 25% sample with 1 and 2
  cores, always on the **validation** stage. Each line of `logs\scale_all.log` must say `exit 0`; re-run
  only the failed method (`.\tools\scale.ps1 <method>`).

- [ ] **C2. Commit** the new `results/*/user_time/val__scale_*` folders on `main`
  (`git add results`, `git commit -m "Scalability runs (RQ3), reference machine"`). Do not push yet.

## Part D: integration (when all four branches are on GitHub)

- [ ] **D1. Are all branches there?** `git fetch origin` then `git branch -r`. Needed:
  `origin/member/knn`, `origin/member/lsh`, `origin/member/ppr`, `origin/member/als`. If one is missing,
  **STOP** and note who is still outstanding. Parts B and C do not depend on this.

- [ ] **D2. Merge.** On `main`, for each of the four: `git merge --no-ff origin/member/<method>`.
  Members only touch their own files, so there should be no conflicts. If there is one, or a branch
  changes anything outside its own files (`git diff --stat main...origin/member/<method>`), **STOP**.

- [ ] **D3. Verify.** `.\.venv\Scripts\python.exe tools\verify_results.py` → `ALL RESULTS COMPARABLE`
  (chosen parameters, full test stage, same evaluated users as the baseline on both splits). A `FAIL`
  goes back to that member; do not patch their numbers.

- [ ] **D4. Tables.**
  ```
  .\.venv\Scripts\python.exe tools\rebuild_summary.py
  .\.venv\Scripts\python.exe tools\summarize.py
  ```
  `reports/results.md` now has tuning, test results with intervals, and scalability.

- [ ] **D5. Paired comparisons** on both splits, with `tools\compare.py` through `run.ps1`: LSH − kNN,
  ALS − kNN, PPR − kNN, and the best method against each other one. Write the differences and 95% intervals
  into `report/comparisons.md`, one table per split.

- [ ] **D6. Scalability section.** Write `tools/plots.py` (matplotlib; add it to `requirements.txt`) that
  reads the `val__scale_*` `metrics.json` files and saves two figures to `report/figures/`: recommend time
  against user sample (4 cores) and against cores (25% sample), one line per method. Write
  `report/scalability.md` around them: the two tables from `reports/results.md`, speed-up 1→4 cores,
  and the LSH against exact kNN time and neighbour-recall trade-off (recall numbers are in `report/lsh.md`).
  State clearly: all scalability timings come from this one machine; the "recommend seconds" in the
  test-result tables come from each member's own machine and must not be compared across methods.

- [ ] **D7. Assemble `report/report.md`** from the sections, in this order: `data.md`, `evaluation.md`,
  `knn.md`, `lsh.md`, `ppr.md`, `als.md`, `comparisons.md`, `scalability.md`, then a short conclusion that
  answers each research question in the proposal with the numbers. Make wording and table formats
  consistent; do not change any number.

- [ ] **D8. Update the trackers.** Tick the finished items in `CHECKLIST.md` and replace the "Status"
  section of `README.md` with the real state.

- [ ] **D9. Push `main`.** Commit, review (rule 5), then `git push origin main`.

- [ ] **D10. Done.** Share with the group: the final results table, where the report is, and anything unresolved.
