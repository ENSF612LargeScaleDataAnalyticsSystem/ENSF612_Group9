# Group workflow: who does what, and how the work comes back together

Tuning is finished. What is left is the final test runs, the scalability runs and the report.
Each member does their part on their own machine and branch, and pushes it to GitHub. One person
(the integrator) merges the four branches and builds the final tables.

| Task file | Method | Also writes | Branch |
|---|---|---|---|
| `tasks/knn.md` | exact user-kNN | integration, scalability, final report (integrator) | `member/knn`, `main` |
| `tasks/lsh.md` | MinHash-LSH | LSH against exact kNN | `member/lsh` |
| `tasks/ppr.md` | Monte Carlo PPR | report section on data and splits | `member/ppr` |
| `tasks/als.md` | ALS | report section on the evaluation protocol | `member/als` |

## What each member does

1. Install [Git](https://git-scm.com/downloads) and make sure your GitHub account can write to the
   group repository.
2. Clone the repository into a folder **outside OneDrive** (for example `C:\ensf612`).
3. Read `CONTRIBUTING.md`, then open your task file and work through it from the first unticked step.
   Tick each step when it is verified, so you can stop and pick up again at any point.
4. At the end, open the pull-request link printed by the push command and press **Create pull request**.
   Do not merge; the integrator merges.

Expect about an hour of setup (downloads, ingest, split, checks) and then the test runs, which take from
half an hour to a few hours depending on the method and the laptop. The laptop must stay awake and
plugged in during the runs. You need about 10 GB of free disk space and ideally 16 GB of RAM (8 GB works
with `--driver-memory 5g`).

## Why the results stay comparable

- **Same rows everywhere.** The splits are deterministic. `tools/member.py check` compares each
  machine's data profile and split statistics with the group's copies and refuses to go on if they differ.
- **Same numbers everywhere.** `tools/member.py pilot` re-runs the chosen settings on the 10% pilot
  (validation stage) and must reproduce the group's metrics before any test run.
- **No tuning on the test part.** Parameters are fixed in `tools/chosen.json`. `tools/member.py test`
  reads them from there, runs each split once and refuses to run it again.
- **No merge conflicts.** Each member commits only their own files (`tools/member.py push` stages nothing
  else, and refuses to push if shared code was changed). `results/summary.csv` is not in git; it is rebuilt
  from the `metrics.json` files by `tools/rebuild_summary.py`.
- **Checked again at the end.** `tools/verify_results.py` confirms that every method used the chosen
  parameters and evaluated exactly the same users as the baseline.
- **Timings.** Run times from different laptops are not comparable, so all scalability runs are made on
  one machine by the integrator.
