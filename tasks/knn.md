# Task: exact user-kNN (`knn`)

These are the steps for the group member who owns **knn** and who published the shared foundation.
Tuning was done on this member's machine. Integration is a separate task: `tasks/integrate.md`.

Do the steps in order. After a step is done **and verified**, change its `- [ ]` to `- [x]` in this file
and go on, so the work can always be picked up again at the first unticked step. Do not stop between
steps unless a step says **STOP**.

Run commands from the project root. Spark runs go through `.\tools\run.ps1 <logname> <python args>` or
`.\.venv\Scripts\python.exe tools\member.py ...` (both set Java/Hadoop).

## Rules

1. **The test stage runs once per method and split**, with the parameters in `tools/chosen.json`. Never
   re-run a test that produced a `metrics.json`. Use `tools/member.py test`, which enforces this.
2. Shared code (`src/`, `tests/`, `tools/`, `requirements.txt`) changes only for a real bug, on `main`,
   with a message to the group so that everybody pulls before testing.
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

- [ ] **B7. Hand over.** Tell the integrator that `member/knn` is on GitHub (`tasks/integrate.md`).
