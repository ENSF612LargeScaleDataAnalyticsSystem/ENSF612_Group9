"""Monte Carlo PPR against exact PPR (RQ2): how close do W walkers get, and at what cost?

    python tools/ppr_exact.py                                   # user_time, 2,000 users, W = 100 250 1000 2500
    python tools/ppr_exact.py --split user_time_10pct --walkers 100 1000

Uses the training part of the validation stage of the split (as tuning and tools/lsh_recall.py do;
the test part is never read) and a fixed random sample of users with at least one like (common.SEED).

The walk is the one in src/methods/ppr.py: n_steps steps, each "user -> random liked movie (visit
counted) -> random user who liked it", then back to the start with probability alpha. Its expected
visit counts can be computed exactly. With x_0 = e_u and P_UM, P_MU the two transition matrices,

    x_{t+1} = alpha * e_u + (1 - alpha) * x_t P_UM P_MU,     E[visits] = n_walkers * sum_t x_t P_UM

(t = 0 .. n_steps - 1). These exact scores are the ground truth. For each W the Monte Carlo counts
come from src.methods.ppr.walk_counts with the method's own seeding, on fixed batches of the sampled
users. Two comparisons:

  * raw PPR (beta 0, min_visits 1): exact expected visits vs Monte Carlo counts. This is the
    approximation itself.
  * the chosen configuration (tools/chosen.json): the same popularity penalty beta on both sides;
    min_visits is a noise filter for Monte Carlo counts and is applied to the Monte Carlo side only.

Reported per W: overlap@10 with the exact top-10 (unseen movies only, as in the method), NDCG@10 of
both on the sampled users against the validation target (shared src/evaluate.py), and seconds per
1,000 users for the Monte Carlo walks and for the exact computation (numpy on the driver, one core,
same machine). Writes reports/ppr_exact.md and reports/ppr_exact.json.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import REPORTS, SEED, get_spark, load_stage  # noqa: E402
from src.evaluate import evaluate, ground_truth  # noqa: E402
from src.methods._shared import build_index, row, top_k  # noqa: E402
from src.methods.ppr import walk_counts  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from member import chosen_params  # noqa: E402

BATCH = 256


def transition_matrices(d):
    """Sparse P_UM (user -> liked movie) and P_MU (movie -> user who liked it), row-stochastic."""
    from scipy import sparse
    n_users = len(d["liked_ptr"]) - 1
    lp, li, kp, ki = d["liked_ptr"], d["liked_idx"], d["likers_ptr"], d["likers_idx"]
    deg_u = np.diff(lp).astype(np.float64)
    deg_m = np.diff(kp).astype(np.float64)
    rows_u = np.repeat(np.arange(n_users), np.diff(lp))
    p_um = sparse.csr_matrix((1.0 / deg_u[rows_u], (rows_u, li)), shape=(n_users, d["n_items"]))
    rows_m = np.repeat(np.arange(d["n_items"]), np.diff(kp))
    p_mu = sparse.csr_matrix((1.0 / deg_m[rows_m], (rows_m, ki)), shape=(d["n_items"], n_users))
    return p_um, p_mu


def exact_counts(batch, p_um, p_mu, alpha, n_walkers, n_steps):
    """(len(batch), n_items) expected visit counts of n_walkers walkers from each user in batch."""
    n_users = p_um.shape[0]
    e = np.zeros((len(batch), n_users))
    e[np.arange(len(batch)), batch] = 1.0
    x = e.copy()
    visits = np.zeros((len(batch), p_um.shape[1]))
    for _ in range(n_steps):
        m = (p_um.T @ x.T).T                     # distribution over the movie visited at this step
        visits += m
        x = alpha * e + (1 - alpha) * (p_mu.T @ m.T).T
    return n_walkers * visits


def top_lists(scores, users, d, k):
    out = {}
    for j, u in enumerate(users):
        out[int(u)] = [int(i) for i in top_k(scores[j], row(d["rated_ptr"], d["rated_idx"], u), k, d["pop_order"])]
    return out


def overlap(a, b, k):
    return float(np.mean([len(set(a[u][:k]) & set(b[u][:k])) / k for u in a]))


def ndcg(spark, lists, truth, train, k):
    rows = [(u, i, r + 1) for u, items in lists.items() for r, i in enumerate(items)]
    recs = spark.createDataFrame(rows, "user INT, item INT, rank INT")
    return evaluate(recs, truth, train=train, k=k, n_boot=0)[f"ndcg@{k}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="user_time")
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--walkers", type=int, nargs="+", default=[100, 250, 1000, 2500])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--driver-memory", default="8g")
    args = ap.parse_args()

    _, chosen = chosen_params("ppr")
    alpha, n_steps = float(chosen.get("alpha", 0.3)), int(chosen.get("n_steps", 20))
    beta, min_visits = float(chosen.get("beta", 0.0)), int(chosen.get("min_visits", 1))
    w_chosen = int(chosen.get("n_walkers", 100))
    walkers = sorted(set(args.walkers) | {w_chosen})

    spark = get_spark("ppr-exact", driver_memory=args.driver_memory)
    try:
        train, target = load_stage(spark, args.split, "val")
        train = train.cache()
        d = build_index(train)
        n_users = len(d["liked_ptr"]) - 1
        has_like = np.flatnonzero(np.diff(d["liked_ptr"]) > 0)
        users = np.intersect1d(d["users"], has_like)
        rng = np.random.default_rng(SEED)
        sample = np.sort(rng.choice(users, size=min(args.users, len(users)), replace=False))
        truth = ground_truth(train, target).join(
            spark.createDataFrame([(int(u),) for u in sample], "user INT"), "user", "left_semi").cache()
        penalty = np.maximum(d["n_likes"], 1).astype(np.float64) ** -beta if beta else None
        batches = [sample[i:i + BATCH] for i in range(0, len(sample), BATCH)]
        print(f"{args.split} (val stage), {len(sample):,} sampled users of {n_users:,}; "
              f"alpha {alpha}, n_steps {n_steps}; chosen: {w_chosen} walkers, beta {beta}, min_visits {min_visits}")

        t0 = time.time()
        p_um, p_mu = transition_matrices(d)
        ex = np.vstack([exact_counts(b, p_um, p_mu, alpha, 1, n_steps) for b in batches])
        t_exact = time.time() - t0
        ex_raw = top_lists(ex, sample, d, args.k)
        ex_pen = top_lists(ex * penalty[None, :], sample, d, args.k) if beta else ex_raw
        res = {"split": args.split, "stage": "val", "sampled_users": int(len(sample)),
               "alpha": alpha, "n_steps": n_steps, "beta": beta, "min_visits": min_visits,
               "chosen_walkers": w_chosen, "machine": platform.platform(),
               "exact": {"seconds": round(t_exact, 2),
                         "seconds_per_1000_users": round(1000 * t_exact / len(sample), 2),
                         "ndcg_raw": ndcg(spark, ex_raw, truth, train, args.k),
                         "ndcg_beta": ndcg(spark, ex_pen, truth, train, args.k)},
               "monte_carlo": []}
        print(f"exact: {t_exact:.1f}s, NDCG@{args.k} raw {res['exact']['ndcg_raw']:.4f}, "
              f"with beta {res['exact']['ndcg_beta']:.4f}")

        for W in walkers:
            t1 = time.time()
            mc = np.vstack([walk_counts(b, d, alpha, W, n_steps, np.random.default_rng([SEED, int(b[0])]))
                            for b in batches])
            t_mc = time.time() - t1
            mc_raw = top_lists(mc, sample, d, args.k)
            filt = mc.copy()
            if min_visits > 1:
                filt[filt < min_visits] = 0
            mc_pen = top_lists(filt * penalty[None, :], sample, d, args.k) if beta else top_lists(filt, sample, d, args.k)
            r = {"walkers": W, "seconds": round(t_mc, 2), "seconds_per_1000_users": round(1000 * t_mc / len(sample), 2),
                 "overlap_raw": round(overlap(ex_raw, mc_raw, args.k), 4),
                 "overlap_chosen_rules": round(overlap(ex_pen, mc_pen, args.k), 4),
                 "ndcg_raw": ndcg(spark, mc_raw, truth, train, args.k),
                 "ndcg_chosen_rules": ndcg(spark, mc_pen, truth, train, args.k)}
            res["monte_carlo"].append(r)
            print(f"  W={W:<5} overlap@{args.k} raw {r['overlap_raw']:.3f} (chosen rules {r['overlap_chosen_rules']:.3f}), "
                  f"NDCG@{args.k} raw {r['ndcg_raw']:.4f} (chosen rules {r['ndcg_chosen_rules']:.4f}), {t_mc:.1f}s")
    finally:
        spark.stop()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "ppr_exact.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    e = res["exact"]
    lines = ["# Monte Carlo PPR against exact PPR (generated by tools/ppr_exact.py)", "",
             f"`{res['split']}` validation stage (training part only), {res['sampled_users']:,} sampled users with at "
             f"least one like, alpha {alpha}, {n_steps} steps, top-{args.k} unseen movies. Machine: {res['machine']}.", "",
             f"Exact expected visits: {e['seconds']} s for the sample ({e['seconds_per_1000_users']} s per 1,000 users), "
             f"NDCG@{args.k} {e['ndcg_raw']:.4f} without and {e['ndcg_beta']:.4f} with the popularity penalty "
             f"(beta {beta}).", "",
             f"| Walkers W | Overlap@{args.k}, raw PPR | NDCG@{args.k}, raw PPR | Overlap@{args.k}, chosen rules | "
             f"NDCG@{args.k}, chosen rules | Monte Carlo s per 1,000 users |", "|---|---|---|---|---|---|"]
    for r in res["monte_carlo"]:
        mark = " (chosen)" if r["walkers"] == w_chosen else ""
        lines.append(f"| {r['walkers']}{mark} | {r['overlap_raw']:.4f} | {r['ndcg_raw']:.4f} | {r['overlap_chosen_rules']:.4f} | "
                     f"{r['ndcg_chosen_rules']:.4f} | {r['seconds_per_1000_users']} |")
    lines += ["", f"Raw PPR: beta 0, min_visits 1. Chosen rules: beta {beta} on both sides, min_visits {min_visits} on the "
              "Monte Carlo side only (it filters sampling noise, which the exact scores do not have)."]
    (REPORTS / "ppr_exact.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("-> reports/ppr_exact.md")


if __name__ == "__main__":
    main()
