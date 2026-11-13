"""Scalability figures (RQ3) from the val__scale_* runs.

    python tools/plots.py

Reads results/<method>/user_time/val__scale_c<cores>[_s<pct>]/metrics.json and writes
  report/figures/time_vs_sample.png   recommend time against user sample, 4 cores
  report/figures/time_vs_cores.png    recommend time against cores, 25% sample
one line per method. All scalability runs come from one machine, so the times can be compared.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
METHODS = ["popularity", "knn", "lsh", "ppr", "als"]
SAMPLES = [10, 25, 50, 100]
CORES = [1, 2, 4]


def seconds(method, cores, pct):
    tag = f"scale_c{cores}" + (f"_s{pct:g}" if pct < 100 else "")
    path = RESULTS / method / "user_time" / f"val__{tag}" / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["seconds_recommend"]


def plot(xs, ys_by_method, xlabel, title, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in METHODS:
        pts = [(x, y) for x, y in zip(xs, ys_by_method[m]) if y is not None]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=m)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("recommend time (s)")
    ax.set_xticks(xs)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot(SAMPLES, {m: [seconds(m, 4, p) for p in SAMPLES] for m in METHODS},
         "users in the sample (%)", "Recommend time against data size (4 cores)",
         FIGURES / "time_vs_sample.png")
    plot(CORES, {m: [seconds(m, c, 25) for c in CORES] for m in METHODS},
         "cores", "Recommend time against cores (25% sample)",
         FIGURES / "time_vs_cores.png")


if __name__ == "__main__":
    main()
