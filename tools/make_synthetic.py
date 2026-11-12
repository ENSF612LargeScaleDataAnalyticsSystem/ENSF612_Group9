"""Generate a small fake dataset in the exact MovieLens 20M file format.

Use it to test the pipeline in seconds without the real 20M files:

    python tools/make_synthetic.py --out data/raw/synthetic --users 2000 --movies 800 --dirty

With --dirty, a few broken rows are injected (null field, unparseable value, duplicate pair,
invalid rating, out-of-range timestamp, rating for a movie missing from movies.csv), and some
movies get no ratings, so you can see that ingest.py detects and reports each case.
"""
import argparse
import csv
import os
import random

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/synthetic")
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--movies", type=int, default=800)
    ap.add_argument("--seed", type=int, default=612)
    ap.add_argument("--dirty", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Movie ids are sparse like the real data (not 1..M).
    movie_ids = np.sort(rng.choice(np.arange(1, args.movies * 5), size=args.movies, replace=False))
    n_unrated = max(3, args.movies // 50) if args.dirty else 0
    rated_movies = movie_ids[: args.movies - n_unrated]

    # Zipf-like popularity so a few movies dominate, as in MovieLens.
    pop = 1.0 / np.arange(1, len(rated_movies) + 1) ** 0.9
    pop = pop / pop.sum()
    quality = rng.normal(0, 0.6, len(rated_movies))

    t_start, t_end = 789_652_009, 1_427_784_002  # 1995-01-09 .. 2015-03-31, as in ml-20m
    rows = []
    for u in range(1, args.users + 1):
        n = int(min(len(rated_movies), 20 + rng.geometric(1 / 60)))
        items = rng.choice(len(rated_movies), size=n, replace=False, p=pop)
        bias = rng.normal(0, 0.5)
        first = rng.integers(t_start, t_end - 86_400 * 30)
        span = rng.integers(3_600, 86_400 * 365 * 3)
        ts = np.sort(rng.integers(first, min(t_end, first + span), size=n))
        # Some users rate several movies in one session with the same timestamp.
        if n > 25 and rng.random() < 0.3:
            ts[5:10] = ts[5]
        for idx, t in zip(items, ts):
            r = 3.4 + bias + quality[idx] + rng.normal(0, 0.8)
            r = float(np.clip(np.round(r * 2) / 2, 0.5, 5.0))
            rows.append([u, int(rated_movies[idx]), f"{r:.1f}", int(t)])

    if args.dirty:
        rows.append([5, int(rated_movies[0]), "", 1_000_000_000])              # missing rating
        rows.append([6, int(rated_movies[1]), "4.0", "not_a_time"])           # unparseable ts
        rows.append(list(rows[10]))                                            # exact duplicate
        dup = list(rows[20]); dup[2] = "1.0"; dup[3] = int(dup[3]) + 99
        rows.append(dup)                                                       # same pair, new rating
        rows.append([7, int(rated_movies[2]), "4.3", 1_000_000_000])          # not a half-star
        rows.append([8, int(rated_movies[3]), "7.0", 1_000_000_000])          # out of range
        rows.append([9, int(rated_movies[4]), "3.0", 100])                    # timestamp in 1970
        rows.append([10, int(movie_ids.max()) + 1000, "4.0", 1_000_000_000])  # orphan movie

    random.shuffle(rows)
    with open(os.path.join(args.out, "ratings.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        w.writerows(rows)

    genres = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Sci-Fi", "Animation"]
    with open(os.path.join(args.out, "movies.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["movieId", "title", "genres"])
        for m in movie_ids:
            g = "|".join(sorted(random.sample(genres, random.randint(1, 3))))
            year = random.randint(1950, 2014)
            # Real titles contain commas and double quotes, which tests the CSV options.
            title = f'"Quoted" Movie {int(m)} ({year})' if random.random() < 0.1 else f"Movie {int(m)}, The ({year})"
            w.writerow([int(m), title, g])

    print(f"wrote {len(rows):,} ratings for {args.users:,} users and {len(movie_ids):,} movies to {args.out}")


if __name__ == "__main__":
    main()
