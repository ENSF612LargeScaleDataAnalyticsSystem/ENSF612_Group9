# Task: scalability runs (RQ3)

These are the steps for the group member who makes the scalability runs. One person makes **all 30 runs
on one machine**, because run times from different machines cannot be compared. It can be anyone who
has already finished the setup for their own method (the data and splits are the same).

Do the steps in order. After a step is done **and verified**, change its `- [ ]` to `- [x]` in this file
and go on, so the work can always be picked up again at the first unticked step. Do not stop between
steps unless a step says **STOP**.

Run every command from the project root with the project's venv Python: `.venv\Scripts\python.exe`
on Windows, `.venv/bin/python` on macOS/Linux (written as `python` below).

## What is measured

Every method (popularity, kNN, LSH, PPR, ALS) with its chosen parameters, on the `user_time`
**validation** stage (the test part is not used):

- **Data size:** 10%, 25%, 50% and 100% of users, with 4 cores.
- **Cores:** 1 and 2 cores on the 25% sample (4 cores comes from the first set).

That is 6 runs per method, 30 in total. Expect several hours on a 4-core laptop, less on a faster machine.

## Rules

1. **Never edit** `src/`, `tests/`, `tools/`, `requirements.txt` or another member's files. Your files are
   the `results/*/user_time/val__scale_*` folders and `tasks/scale.md`.
2. **One machine, nothing else running.** Close other heavy programs, keep the laptop plugged in and
   awake, and do not use it for other work during the runs. The times are the result.
3. **Same settings for every run.** Do not change `--driver-memory` between runs. If the machine has less
   than 16 GB of RAM, use `--driver-memory 5g` for all 30 runs from the start.
4. A run that crashes is simply repeated; `tools/scale.py` skips runs that already have a result.
   Never delete or repeat a finished run to get a "better" time.

## Steps

- [ ] **1. Branch.** The machine must already have passed the setup of a method task (steps 1–7 of
  `tasks/lsh.md`, `ppr.md` or `als.md`). If your own method's work is not pushed yet, finish and push
  that first. Then:
  ```
  git fetch origin
  git checkout -b member/scale origin/main
  ```
  Verify: `git branch --show-current` prints `member/scale`, and `tools/scale.py` exists.

- [ ] **2. Check.** `python tools/member.py check scale` must print `ALL CHECKS PASSED`.

- [ ] **3. See the plan.** `python tools/scale.py --list` shows 30 runs "to do".

- [ ] **4. Run.** Start in the background and wait:
  ```
  python tools/scale.py
  ```
  It must end with `SCALABILITY RUNS COMPLETE`. If it stops with `[FAIL]`, read the end of the log it
  names, fix the cause (memory, disk space, another program) and run the same command again.
  If the same run fails twice, **STOP** and tell the group which one.

- [ ] **5. Sanity check.** `python tools/scale.py --list` shows 30 runs done. For each method, the time
  should grow with the sample size and should not grow when cores are added. Note anything that
  does not, without re-running it.

- [ ] **6. Push.** `python tools/member.py push scale`. Open the printed link in a browser and press
  **Create pull request** (into `main`). Do not merge; the integrator merges.

- [ ] **7. Done.** Post a short message to the group: that the 30 runs are in, the machine they were made
  on (CPU count, RAM, operating system), the driver memory used, and anything odd from step 5.
