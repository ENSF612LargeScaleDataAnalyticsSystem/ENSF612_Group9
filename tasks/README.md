# Group workflow: who does what, and how the work comes back together

Tuning is finished. What is left is the final test runs, the scalability runs and the report.
Each member runs their part with Claude Code in VS Code, on their own branch, and pushes it to
GitHub. One person (the integrator) merges the four branches and builds the final tables.

| Task file | Method | Also writes | Branch |
|---|---|---|---|
| `tasks/knn.md` | exact user-kNN | integration, scalability, final report (integrator) | `member/knn`, `main` |
| `tasks/lsh.md` | MinHash-LSH | LSH against exact kNN | `member/lsh` |
| `tasks/ppr.md` | Monte Carlo PPR | report section on data and splits | `member/ppr` |
| `tasks/als.md` | ALS | report section on the evaluation protocol | `member/als` |

## What each member does

1. Install [Git](https://git-scm.com/downloads) and Claude Code for VS Code, and make sure your GitHub
   account can write to the group repository.
2. Open an **empty folder outside OneDrive** in VS Code (for example `C:\ensf612`), open Claude Code and
   paste your starter prompt below.
3. Whenever Claude stops, read what it says. If it asks for something (install Java, sign in to GitHub),
   do it. Then type **continue**. Repeat until it says the task is done.
4. Open the pull-request link Claude gives you at the end and press **Create pull request**. Do not merge.

Expect about an hour of setup (downloads, ingest, split, checks) and then the test runs, which take from
half an hour to a few hours depending on the method and the laptop. The laptop must stay awake and
plugged in during the runs. You need about 10 GB of free disk space and ideally 16 GB of RAM (8 GB works
with `--driver-memory 5g`; tell Claude your RAM if it does not ask).

## Starter prompts (replace nothing, paste as is)

**LSH**

> I am the LSH member of ENSF 612 Group 9. If this folder does not contain the project yet, clone
> https://github.com/ENSF612LargeScaleDataAnalyticsSystem/ENSF612_Group9.git and work inside it. Read
> CLAUDE.md and tasks/lsh.md, then do the first unticked step and keep going step by step, ticking each
> step in tasks/lsh.md once it is verified. Only stop where a step says STOP or when everything is ticked.

**PPR**

> I am the PPR member of ENSF 612 Group 9. If this folder does not contain the project yet, clone
> https://github.com/ENSF612LargeScaleDataAnalyticsSystem/ENSF612_Group9.git and work inside it. Read
> CLAUDE.md and tasks/ppr.md, then do the first unticked step and keep going step by step, ticking each
> step in tasks/ppr.md once it is verified. Only stop where a step says STOP or when everything is ticked.

**ALS**

> I am the ALS member of ENSF 612 Group 9. If this folder does not contain the project yet, clone
> https://github.com/ENSF612LargeScaleDataAnalyticsSystem/ENSF612_Group9.git and work inside it. Read
> CLAUDE.md and tasks/als.md, then do the first unticked step and keep going step by step, ticking each
> step in tasks/als.md once it is verified. Only stop where a step says STOP or when everything is ticked.

**kNN and integration** (in the existing project folder on the reference machine)

> I am the kNN member and the integrator of ENSF 612 Group 9. Read CLAUDE.md and tasks/knn.md, then do the
> first unticked step and keep going step by step, ticking each step in tasks/knn.md once it is verified.
> Only stop where a step says STOP or when everything is ticked.

After that, in every later session: **continue with my task file**.

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
