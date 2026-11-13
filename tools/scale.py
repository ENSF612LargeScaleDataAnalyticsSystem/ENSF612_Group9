"""Scalability runs (RQ3) on the user_time validation stage; the test part is never touched here.

    python tools/scale.py                 # all methods
    python tools/scale.py knn lsh         # some methods
    python tools/scale.py --list          # show which runs are done and which are left

Same runs as tools/scale.ps1, but works on Windows, macOS and Linux and can be stopped and started
again: a run that already has a metrics.json is skipped.

  1. data size: user samples 10 / 25 / 50 / 100% with 4 cores
  2. cores:     1 / 2 cores on the 25% sample (4 cores comes from step 1)

Every run uses the same --driver-memory and the parameters in tools/chosen.json. All 30 runs must be
made on ONE machine, with nothing else heavy running: the wall-clock times are the result.
Results: results/<method>/user_time/val__scale_c<cores>[_s<pct>]/metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from member import ROOT, chosen_params, run_logged, spark_env  # noqa: E402

METHODS = ("popularity", "knn", "lsh", "ppr", "als")
RUNS = [("4", 0.1), ("4", 0.25), ("4", 0.5), ("4", 1.0), ("1", 0.25), ("2", 0.25)]   # (cores, user sample)


def folder(method, cores, sample):
    tag = f"scale_c{cores}" + (f"_s{sample * 100:g}" if sample < 1 else "")
    return ROOT / "results" / method / "user_time" / f"val__{tag}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("methods", nargs="*", choices=[*METHODS, []], help="default: all five")
    ap.add_argument("--driver-memory", default="8g", help="must be the same for every run (default 8g)")
    ap.add_argument("--list", action="store_true", help="only show the state of each run")
    args = ap.parse_args()
    methods = args.methods or METHODS

    todo = []
    for m in methods:
        for cores, sample in RUNS:
            out = folder(m, cores, sample)
            done = (out / "metrics.json").exists()
            if done:
                j = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
                state = f"done  recommend {j['seconds_recommend']:>8.1f}s  NDCG@10 {j['ndcg@10']:.4f}"
            else:
                state = "to do"
                todo.append((m, cores, sample))
            print(f"  {m:<11} cores {cores}  sample {sample * 100:>3g}%   {state}")
    print(f"\n{len(todo)} of {len(methods) * len(RUNS)} runs left")
    if args.list or not todo:
        return 0

    env = spark_env()
    for m, cores, sample in todo:
        flags, _ = chosen_params(m)
        print(f"\n{m}: cores {cores}, sample {sample * 100:g}%", flush=True)
        code = run_logged(f"scale_{m}_c{cores}_s{sample * 100:g}",
                          ["-m", "src.run", "--method", m, "--split", "user_time", "--stage", "val", "--n-boot", "0",
                           "--cores", cores, "--driver-memory", args.driver_memory, "--sample", str(sample),
                           "--tag", f"scale_c{cores}", *flags], env)
        if code != 0 or not (folder(m, cores, sample) / "metrics.json").exists():
            print(f"[FAIL] exit {code}. Read the end of the log, close other programs, and run this command again:\n"
                  "       finished runs are skipped. A timing run that crashed is simply repeated.")
            return 1
        j = json.loads((folder(m, cores, sample) / "metrics.json").read_text(encoding="utf-8"))
        print(f"[DONE] recommend {j['seconds_recommend']:.1f}s, NDCG@10 {j['ndcg@10']:.4f}", flush=True)
    print("\nSCALABILITY RUNS COMPLETE. Next: python tools/member.py push scale")
    return 0


if __name__ == "__main__":
    sys.exit(main())
