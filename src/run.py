"""Run one method on one split and stage, then evaluate it. Every method goes through this script.

    python -m src.run --method popularity --split user_time_10pct --stage val
    python -m src.run --method popularity --split user_time --stage test
    python -m src.run --method lsh --split user_time_10pct --stage val --param n_hashes=128 --param bands=32 --tag n128_b32

Stages: "val" trains on train and scores on validation (use it for tuning, as often as you like);
"test" trains on train + validation and scores on test (run it once per method, with the chosen
parameters).

Writes results/<method>/<split>/<stage>[__<tag>]/{recs, per_user, metrics.json}
and appends one line to results/summary.csv.
"""
import argparse
import ast
import csv
import importlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

from src.common import K, RESULTS, get_spark, load_stage, user_sample
from src.evaluate import evaluate, ground_truth


def parse_params(items):
    params = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param needs key=value, got {it!r}")
        key, raw = it.split("=", 1)
        try:
            params[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            params[key] = raw
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="module name in src/methods, e.g. popularity")
    ap.add_argument("--split", default="user_time_10pct")
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--param", action="append", help="hyper-parameter key=value (repeatable)")
    ap.add_argument("--tag", default="", help="label for this configuration, used in the output folder")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--cores", default="*")
    ap.add_argument("--driver-memory", default="8g")
    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples; 0 to skip")
    ap.add_argument("--sample", type=float, default=1.0,
                    help="keep this fraction of users (common.user_sample, nested samples) for RQ3; "
                         "adds s<percent> to the tag")
    args = ap.parse_args()

    if not 0 < args.sample <= 1:
        raise SystemExit("--sample must be in (0, 1]")
    if args.sample < 1:
        args.tag = "_".join(t for t in (args.tag, f"s{args.sample * 100:g}") if t)
    params = parse_params(args.param)
    method = importlib.import_module(f"src.methods.{args.method}")
    out = RESULTS / args.method / args.split / (args.stage + (f"__{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    spark = get_spark(f"{args.method}-{args.split}-{args.stage}", cores=args.cores, driver_memory=args.driver_memory)
    try:
        train, target = load_stage(spark, args.split, args.stage)
        if args.sample < 1:
            train, target = user_sample(train, args.sample), user_sample(target, args.sample)

        t0 = time.time()
        recs = method.recommend(spark, train, k=args.k, **params)
        recs.select("user", "item", "rank").write.mode("overwrite").parquet(str(out / "recs"))
        t_rec = time.time() - t0

        t1 = time.time()
        recs = spark.read.parquet(str(out / "recs"))
        metrics = evaluate(recs, ground_truth(train, target), train=train, k=args.k,
                           n_boot=args.n_boot, per_user_path=str(out / "per_user"))
        t_eval = time.time() - t1

        info = {
            "method": args.method, "split": args.split, "stage": args.stage, "tag": args.tag,
            "params": params, "sample": args.sample, "seconds_recommend": round(t_rec, 2), "seconds_evaluate": round(t_eval, 2),
            **metrics,
            "env": {"spark": spark.version, "python": platform.python_version(),
                    "platform": platform.platform(), "machine": platform.machine(),
                    "cpu_count": os.cpu_count(), "cores": args.cores, "driver_memory": args.driver_memory},
            "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "command": " ".join(sys.argv),
        }
        (out / "metrics.json").write_text(json.dumps(info, indent=2))
        append_summary(info)

        k = args.k
        print(f"\n{args.method} on {args.split} ({args.stage}), {metrics['evaluated_users']:,} users")
        for m in ("precision", "recall", "ndcg"):
            ci = metrics.get(f"{m}@{k}_ci95")
            ci_txt = f"  [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
            print(f"  {m + '@' + str(k):<14} {metrics[f'{m}@{k}']:.4f}{ci_txt}")
        print(f"  {'coverage':<14} {metrics['coverage']:.4f}")
        print(f"  time: recommend {t_rec:.1f}s, evaluate {t_eval:.1f}s -> {out}")
    finally:
        spark.stop()


def append_summary(info):
    path = RESULTS / "summary.csv"
    k = info["k"]
    row = {
        "finished_utc": info["finished_utc"], "method": info["method"], "split": info["split"],
        "stage": info["stage"], "tag": info["tag"], "params": json.dumps(info["params"]),
        f"precision@{k}": info[f"precision@{k}"], f"recall@{k}": info[f"recall@{k}"],
        f"ndcg@{k}": info[f"ndcg@{k}"], "coverage": info["coverage"],
        "evaluated_users": info["evaluated_users"], "seconds_recommend": info["seconds_recommend"],
        "cores": info["env"]["cores"],
    }
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
