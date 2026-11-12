"""Write tools/chosen.json: for each method the parameters with the best NDCG@10 on the pilot validation stage.

    python tools/pick.py

LSH only considers settings with at least LSH_MIN_ROWS rows per band (see below).
LSH is meant to approximate exact kNN, so it takes the banding (n_hashes, bands, max_bucket) from its
own best run but the neighbour settings (similarity, n_neighbours) from kNN's best run. Then the two
differ only in how neighbours are found.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LSH_FROM_KNN = ("similarity", "n_neighbours")
# r = 1 row per band makes any shared liked movie a likely candidate: on the pilot that is ~6,000 of 13,600
# users per user and 5x slower than exact kNN; on the full data it would not finish. It is reported
# in the tuning table but not chosen.
LSH_MIN_ROWS = 2


def rows_per_band(p):
    return int(p.get("n_hashes", 128)) // int(p.get("bands", 32))     # lsh.py defaults


def main():
    rows = list(csv.DictReader(open(ROOT / "results" / "summary.csv", encoding="utf-8")))
    rows = [r for r in rows if r["split"] == "user_time_10pct" and r["stage"] == "val"]
    best = {}
    for r in rows:
        m = r["method"]
        if m == "lsh" and rows_per_band(json.loads(r["params"])) < LSH_MIN_ROWS:
            continue
        if m not in best or float(r["ndcg@10"]) > float(best[m]["ndcg@10"]):
            best[m] = r
    params = {m: json.loads(r["params"]) for m, r in best.items()}
    if "lsh" in params and "knn" in params:
        for key in LSH_FROM_KNN:
            params["lsh"].pop(key, None)
            if key in params["knn"]:
                params["lsh"][key] = params["knn"][key]
    chosen = {m: [f"{k}={v}" for k, v in p.items()] for m, p in params.items() if m != "popularity"}
    (ROOT / "tools" / "chosen.json").write_text(json.dumps(chosen, indent=2))
    for m, r in best.items():
        print(f"{m:<11} best val NDCG@10 {float(r['ndcg@10']):.4f} (tag {r['tag'] or '-'}) -> {chosen.get(m, [])}")


if __name__ == "__main__":
    main()
