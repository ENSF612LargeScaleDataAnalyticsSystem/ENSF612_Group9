"""Check that the final test results of all methods are comparable before they go into the report.

    python tools/verify_results.py            # all methods
    python tools/verify_results.py lsh als    # some methods

For every method and split (user_time, global_time) it checks that
  * results/<method>/<split>/test/metrics.json and per_user/ exist;
  * the run used exactly the parameters in tools/chosen.json (chosen on validation);
  * it was the full test stage (no user sample);
  * it evaluated the same number of users as the popularity baseline on that split, which only
    happens when the member's machine held the same train / val / test rows.
Exit code 1 if anything is missing or wrong.
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLITS = ("user_time", "global_time")


def chosen(method):
    out = {}
    for item in json.loads((ROOT / "tools" / "chosen.json").read_text(encoding="utf-8")).get(method, []):
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def load(method, split):
    p = RESULTS / method / split / "test" / "metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    methods = sys.argv[1:] or ["knn", "lsh", "ppr", "als"]
    bad = 0
    for split in SPLITS:
        base = load("popularity", split)
        print(f"\n{split}: popularity evaluated {base['evaluated_users']:,} users")
        for m in methods:
            j = load(m, split)
            problems = []
            if j is None:
                problems.append("no metrics.json (not run yet)")
            else:
                if j["params"] != chosen(m):
                    problems.append(f"parameters {j['params']} are not the chosen ones {chosen(m)}")
                if j["stage"] != "test" or j.get("sample", 1.0) != 1.0 or j["tag"]:
                    problems.append("not a plain full test run")
                if j["evaluated_users"] != base["evaluated_users"]:
                    problems.append(f"evaluated {j['evaluated_users']:,} users, not {base['evaluated_users']:,}: different split")
                if not any((RESULTS / m / split / "test" / "per_user").glob("*.parquet")):
                    problems.append("per_user/ is missing (needed for paired comparisons)")
            bad += bool(problems)
            if problems:
                print(f"  [FAIL] {m}: " + "; ".join(problems))
            else:
                env = j["env"]
                print(f"  [ OK ] {m}: NDCG@10 {j['ndcg@10']:.4f}, recommend {j['seconds_recommend']:.0f}s on "
                      f"{env['cpu_count']} CPUs ({env['platform']}), finished {j['finished_utc']} UTC")
    print("\nALL RESULTS COMPARABLE" if not bad else f"\n{bad} PROBLEM(S)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
