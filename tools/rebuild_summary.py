"""Rebuild results/summary.csv from every results/<method>/<split>/<run>/metrics.json.

    python tools/rebuild_summary.py

summary.csv is appended to by every run, so four members committing it would always conflict.
It is therefore not in git: each member commits only results/<method>/, and this script puts the
table back together after the branches are merged. Run it before tools/summarize.py.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    rows = []
    for path in RESULTS.glob("*/*/*/metrics.json"):
        j = json.loads(path.read_text(encoding="utf-8"))
        k = j["k"]
        rows.append({
            "finished_utc": j["finished_utc"], "method": j["method"], "split": j["split"],
            "stage": j["stage"], "tag": j["tag"], "params": json.dumps(j["params"]),
            f"precision@{k}": j[f"precision@{k}"], f"recall@{k}": j[f"recall@{k}"],
            f"ndcg@{k}": j[f"ndcg@{k}"], "coverage": j["coverage"],
            "evaluated_users": j["evaluated_users"], "seconds_recommend": j["seconds_recommend"],
            "cores": j["env"]["cores"],
        })
    if not rows:
        raise SystemExit("no metrics.json files under results/")
    rows.sort(key=lambda r: (r["finished_utc"], r["method"], r["split"], r["tag"]))
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {RESULTS / 'summary.csv'}: {len(rows)} runs")


if __name__ == "__main__":
    main()
