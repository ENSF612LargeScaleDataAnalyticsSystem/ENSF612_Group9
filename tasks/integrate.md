# Task: integration and final report

These are the steps for the **integrator**: the one person who merges the five branches into `main` and
builds the final tables and report. Any member can take this role; it needs a machine where the project
is set up (steps 1-7 of a method task), write access to `main`, and no heavy computing: the longest
commands are the paired comparisons, a few minutes each.

Do the steps in order. After a step is done **and verified**, change its `- [ ]` to `- [x]` in this file
and go on, so the work can always be picked up again at the first unticked step. Do not stop between
steps unless a step says **STOP**.

Run every command from the project root with the project's venv Python (written as `python` below).
On Windows, Spark commands can also go through `.	ools
un.ps1 <logname> <python args>`.

## Rules

1. Work on `main`, starting from the latest `origin/main` (`git checkout main`, `git pull origin main`).
   Never force-push.
2. Do not change any member's numbers. A result that fails a check goes back to that member.
3. Every number in a report file is copied from a `metrics.json` or a log.
4. Pushing `main` publishes to the group: review what will be pushed first
   (`git status`, `git log --oneline origin/main..HEAD`).
5. One Spark job at a time.

## Steps

- [x] **1. Are all branches there?** `git fetch origin` then `git branch -r`. Needed:
  `origin/member/knn`, `origin/member/lsh`, `origin/member/ppr`, `origin/member/als`, `origin/member/scale`. If one is missing,
  **STOP** and note who is still outstanding.

- [ ] **2. Merge.** On `main`, for each of the five: `git merge --no-ff origin/member/<name>`.
  Members only touch their own files. The one expected conflict is in a member's own `tasks/<name>.md`
  (its wording changed on `main` after some members had started): keep the member's version
  (`git checkout --theirs tasks/<name>.md`, `git add` it, `git commit`). For any other conflict, or if a
  branch changes anything outside its own files (`git diff --stat main...origin/member/<name>`), **STOP**.

- [ ] **3. Verify.** `python tools/verify_results.py` → `ALL RESULTS COMPARABLE`
  (chosen parameters, full test stage, same evaluated users as the baseline on both splits). A `FAIL`
  goes back to that member; do not patch their numbers.

- [ ] **4. Tables.**
  ```
  python tools/rebuild_summary.py
  python tools/summarize.py
  ```
  `reports/results.md` now has tuning, test results with intervals, and scalability.

- [ ] **5. Paired comparisons** on both splits, with `python tools/compare.py <run folder A> <run folder B>`: LSH − kNN,
  ALS − kNN, PPR − kNN, and the best method against each other one. Write the differences and 95% intervals
  into `report/comparisons.md`, one table per split.

- [ ] **6. Scalability section.** Write `tools/plots.py` (matplotlib; add it to `requirements.txt`) that
  reads the `val__scale_*` `metrics.json` files and saves two figures to `report/figures/`: recommend time
  against user sample (4 cores) and against cores (25% sample), one line per method. Write
  `report/scalability.md` around them: the two tables from `reports/results.md`, speed-up 1→4 cores,
  and the LSH against exact kNN time and neighbour-recall trade-off (recall numbers are in `report/lsh.md`).
  State clearly: all scalability timings come from one machine (name it, from `env` in the `metrics.json`
  files); the "recommend seconds" in the test-result tables come from each member's own machine and must
  not be compared across methods.

- [ ] **7. Assemble `report/report.md`** from the sections, in this order: `data.md`, `evaluation.md`,
  `knn.md`, `lsh.md`, `ppr.md`, `als.md`, `comparisons.md`, `scalability.md`, then a short conclusion that
  answers each research question in the proposal with the numbers. Make wording and table formats
  consistent; do not change any number.

- [ ] **8. Update the trackers.** Tick the finished items in `CHECKLIST.md` and replace the "Status"
  section of `README.md` with the real state.

- [ ] **9. Push `main`.** Commit, review (rule 4), then `git push origin main`.

- [ ] **10. Done.** Share with the group: the final results table, where the report is, and anything unresolved.
