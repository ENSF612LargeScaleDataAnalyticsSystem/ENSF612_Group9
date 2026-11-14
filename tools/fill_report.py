"""Write the results of tools/ppr_exact.py and tools/sysmetrics.py into report/report.md.

    python tools/ppr_exact.py
    python tools/sysmetrics.py --driver-memory 14g
    python tools/fill_report.py

report/report.md holds four markers (PPR_EXACT_SECTION, PPR_EXACT_CONCLUSION, SYSMETRICS_SECTION,
SYSMETRICS_CONCLUSION). This script replaces each one with text built only from
reports/ppr_exact.json and reports/sysmetrics.json, so every number comes from a run. It refuses to
run if a result file is missing, and it leaves the report unchanged in that case.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report" / "report.md"
NAMES = {"popularity": "popularity", "knn": "exact kNN", "lsh": "LSH", "ppr": "PPR", "als": "ALS"}


def load(name):
    f = ROOT / "reports" / name
    if not f.exists():
        sys.exit(f"{f.relative_to(ROOT)} is missing: run the tool that writes it first")
    return json.loads(f.read_text(encoding="utf-8"))


def ppr_texts(r):
    e, mc = r["exact"], r["monte_carlo"]
    w_c = r["chosen_walkers"]
    c = next(x for x in mc if x["walkers"] == w_c)
    lo, hi = mc[0], mc[-1]
    rows = "\n".join(
        f"| {x['walkers']:,}{' (chosen)' if x['walkers'] == w_c else ''} | {x['overlap_raw']:.4f} | {x['ndcg_raw']:.4f} | "
        f"{x['overlap_chosen_rules']:.4f} | {x['ndcg_chosen_rules']:.4f} | {x['seconds_per_1000_users']} |" for x in mc)
    cheaper = [x["walkers"] for x in mc if x["seconds_per_1000_users"] < e["seconds_per_1000_users"]]
    if not cheaper:
        cost = (f"At every W tested the walks took longer than the exact computation "
                f"({lo['seconds_per_1000_users']} s per 1,000 users at W = {lo['walkers']:,} against "
                f"{e['seconds_per_1000_users']} s for exact PPR), so on this data the Monte Carlo approximation saves no time.")
    elif len(cheaper) == len(mc):
        cost = (f"At every W tested the walks were faster than the exact computation "
                f"({hi['seconds_per_1000_users']} s per 1,000 users at W = {hi['walkers']:,} against "
                f"{e['seconds_per_1000_users']} s for exact PPR).")
    else:
        cost = (f"The walks are cheaper than the exact computation ({e['seconds_per_1000_users']} s per 1,000 users) "
                f"up to W = {max(cheaper):,} and more expensive above it; at the chosen W = {w_c:,} they took "
                f"{c['seconds_per_1000_users']} s per 1,000 users.")
    section = f"""The proposal's ground truth for Monte Carlo PPR is exact PPR on 2,000 sampled users.
`tools/ppr_exact.py` computes the exact expected visit counts of the method's own walk (alpha {r['alpha']},
{r['n_steps']} steps; Section 6.1) by repeated sparse matrix products, and compares the top-10 unseen movies
with those of the Monte Carlo walks from the same users. It uses the training part of the validation stage
of `{r['split']}` ({r['sampled_users']:,} users with at least one like, sampled with seed 612); the test part is
never read. NDCG@10 is scored against the validation target with `src/evaluate.py`.

Two comparisons: **raw PPR** (no popularity penalty, no visit filter) measures the approximation itself;
the **chosen rules** apply the same penalty (beta {r['beta']}) to both sides and the visit filter
(min_visits {r['min_visits']}) to the Monte Carlo side only, because exact scores have no sampling noise to filter.

| Walkers W | Overlap@10, raw | NDCG@10, raw | Overlap@10, chosen rules | NDCG@10, chosen rules | Monte Carlo s per 1,000 users |
|---|---|---|---|---|---|
{rows}

Exact PPR: NDCG@10 {e['ndcg_raw']:.4f} raw and {e['ndcg_beta']:.4f} with the penalty; {e['seconds_per_1000_users']} s per
1,000 users. Exact and Monte Carlo scores are both computed with numpy on one core of the driver, in the same
run on the same machine, so their times can be compared.

Raw overlap with the exact top 10 goes from {lo['overlap_raw']:.4f} at W = {lo['walkers']:,} to {hi['overlap_raw']:.4f} at
W = {hi['walkers']:,}. {cost}"""
    conclusion = (f"For Monte Carlo PPR, the top 10 of the chosen {w_c:,} walkers overlaps the exact PPR top 10 by "
                  f"{c['overlap_raw']:.4f} (raw scores), with NDCG@10 {c['ndcg_raw']:.4f} against {e['ndcg_raw']:.4f} for "
                  f"exact PPR on the same {r['sampled_users']:,} users (Section 6.6). {cost}")
    return section, conclusion


def sys_texts(rows):
    by = {}
    for x in rows:
        by.setdefault(x["method"], {})[x["sample"]] = x
    full = {m: v[max(v)] for m, v in by.items()}
    small = {m: v[min(v)] for m, v in by.items()}
    lo_s, hi_s = min(min(v) for v in by.values()), max(max(v) for v in by.values())

    def mb(b):
        return f"{b / 2**20:,.1f}"

    def gb(b):
        return f"{b / 2**30:.2f}"

    table = "\n".join(
        f"| {NAMES.get(m, m)} | {mb(small[m]['recommend']['shuffle_write_bytes'])} | {mb(full[m]['recommend']['shuffle_write_bytes'])} | "
        f"{mb(full[m]['recommend']['disk_spilled_bytes'])} | {gb(small[m]['peak_rss_bytes'])} | {gb(full[m]['peak_rss_bytes'])} | "
        f"{gb(full[m]['peak_jvm_heap_bytes']) if full[m]['peak_jvm_heap_bytes'] else 'n/a'} |" for m in full)
    any_row = next(iter(full.values()))
    top_shuffle = max(full, key=lambda m: full[m]["recommend"]["shuffle_write_bytes"])
    top_mem = max(full, key=lambda m: full[m]["peak_rss_bytes"])
    low_mem = min(full, key=lambda m: full[m]["peak_rss_bytes"])
    section = f"""The scalability timings above do not include shuffle volume or memory, so `tools/sysmetrics.py` repeats the
size runs ({lo_s * 100:g}% to {hi_s * 100:g}% of the users, 4 cores, chosen parameters, driver memory {any_row['driver_memory']}) on the same machine and
reads Spark's monitoring REST API (shuffle and spill, summed over the stages of the recommend phase) and the
resident memory of the whole process tree (Python driver, Spark JVM, Python workers; sampled every 0.5 s).

| Method | Shuffle write, {lo_s * 100:g}% (MB) | Shuffle write, {hi_s * 100:g}% (MB) | Spilled to disk, {hi_s * 100:g}% (MB) | Peak memory, {lo_s * 100:g}% (GB) | Peak memory, {hi_s * 100:g}% (GB) | Peak JVM heap, {hi_s * 100:g}% (GB) |
|---|---|---|---|---|---|---|
{table}

(Full table with every size in `reports/sysmetrics.md`.) At {hi_s * 100:g}% of the users, {NAMES.get(top_shuffle, top_shuffle)} writes
the most shuffle data ({mb(full[top_shuffle]['recommend']['shuffle_write_bytes'])} MB) and {NAMES.get(top_mem, top_mem)} needs the
most memory ({gb(full[top_mem]['peak_rss_bytes'])} GB, against {gb(full[low_mem]['peak_rss_bytes'])} GB for
{NAMES.get(low_mem, low_mem)}). Exact kNN and PPR collect the training index on the driver and broadcast it to the
workers, so they shuffle almost nothing and their cost shows up as memory; LSH shuffles its (band, bucket) keys
when it groups users into buckets, and MLlib's ALS shuffles factor blocks in every iteration."""
    conclusion = (f"Peak memory at {hi_s * 100:g}% of the users ranges from {gb(full[low_mem]['peak_rss_bytes'])} GB "
                  f"({NAMES.get(low_mem, low_mem)}) to {gb(full[top_mem]['peak_rss_bytes'])} GB ({NAMES.get(top_mem, top_mem)}), "
                  f"and {NAMES.get(top_shuffle, top_shuffle)} shuffles the most data "
                  f"({mb(full[top_shuffle]['recommend']['shuffle_write_bytes'])} MB; Section 9.5).")
    return section, conclusion


def main():
    text = REPORT.read_text(encoding="utf-8")
    markers = ["PPR_EXACT_SECTION", "PPR_EXACT_CONCLUSION", "SYSMETRICS_SECTION", "SYSMETRICS_CONCLUSION"]
    missing = [m for m in markers if m not in text]
    if missing:
        sys.exit(f"markers already replaced or missing: {', '.join(missing)}")
    ppr_s, ppr_c = ppr_texts(load("ppr_exact.json"))
    sys_s, sys_c = sys_texts(load("sysmetrics.json"))
    for m, t in zip(markers, [ppr_s, ppr_c, sys_s, sys_c]):
        text = text.replace(m, t)
    REPORT.write_text(text, encoding="utf-8")
    print("report/report.md updated: Sections 6.6, 9.5 and the RQ2/RQ3 conclusions")


if __name__ == "__main__":
    main()
