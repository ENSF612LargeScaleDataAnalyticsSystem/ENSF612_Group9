# Data and splits

All numbers below are copied from `reports/data_profile.md`, `reports/data_quality.json` and
`reports/split_stats.md`, which `src/ingest.py` and `src/split.py` write. Before a member runs the test
stage, `tools/member.py check` confirms that their machine produced identical copies of these files.

## 1. Dataset

We use MovieLens 20M (GroupLens). After cleaning:

| Statistic | Value |
|---|---|
| Ratings | 20,000,263 |
| Users | 138,493 |
| Movies with at least one rating (in `movies.csv`) | 26,744 (27,278) |
| Density of the user–movie matrix | 0.54% |
| Likes (rating ≥ 4.0) | 9,995,410 (49.98%) |
| Period | 1995-01-09 11:46:44 to 2015-03-31 06:40:02 UTC |
| Ratings per user: min / median / mean / p99 / max | 20 / 68 / 144.41 / 1114 / 9254 |
| Likes per user: min / median / mean / p99 / max | 0 / 37 / 72.17 / 492 / 3177 |
| Ratings per movie: min / median / mean / p99 / max | 1 / 18 / 747.84 / 14396 / 67310 |
| Users without any like | 206 |
| Movies without any like / with only 1 rating | 6024 / 3972 |
| Share of ratings on the top 1% of movies | 33.48% |
| Ratings that share a timestamp with another rating of the same user | 30.9% |

The data is very sparse and skewed: a third of all ratings go to the top 1% of movies, while the
median movie has 18 ratings.

## 2. Cleaning

`src/ingest.py` applies five rules and counts every one of them in `reports/data_quality.json`, even
when the count is 0:

| Rule | Rows removed |
|---|---|
| 1. Missing or unparseable field (userId / movieId / rating / timestamp) | missing 0 / 0 / 0 / 0, unparseable 0 / 0 / 0 / 0 |
| 2. Rating outside 0.5–5.0 or not on a half-star step | 0 |
| 3. Timestamp outside the collection period (9 Jan 1995 – 31 Mar 2015) | 0 |
| 4. `movieId` not in `movies.csv` | 0 |
| 5. Duplicate user–movie pair (latest rating kept) | 0 |

No rating was removed: 20,000,263 rows raw and 20,000,263 clean. The rules stay in the pipeline so the
data is checked, not assumed clean. In `movies.csv`, 534 movies have no rating and get no dense id, so
no method can recommend them. 23 titles have no year and 246 movies have no genres; both are kept,
because the methods use neither titles nor genres. Users and movies are never removed for being
unusual, for example for having very many ratings.

## 3. Likes

A rating of 4.0 or more is a **like**. Likes are the positive signal for every method and for the
evaluation: a recommended movie is a hit if the user liked it in the target part.

## 4. Splits

**`user_time`.** Each user's ratings are sorted by (timestamp, movie id). The last ceil(20% of n)
ratings are test and the ceil(10% of n) before them are validation. The movie id breaks ties, because
30.9% of ratings share a timestamp with another rating of the same user; this makes the split
deterministic. Every user has at least 20 ratings, so every user keeps at least 13 for training.

**`global_time`.** Two timestamp cutoffs shared by all users: the latest 20% of all ratings are test
and the 10% before them are validation. Validation starts at 2007-12-08 01:20:38 UTC and test at
2009-10-20 07:33:43 UTC.

**Pilot samples.** `user_time_10pct` and `global_time_10pct` keep 10% of users with all their ratings.
Users are picked by a hash of their id, not by a random draw, so the samples are nested
(10% ⊂ 25% ⊂ 50%) and the same on every machine. All tuning used the validation stage of
`user_time_10pct`.

| Split | Part | Ratings | Share | Users | Movies | Likes |
|---|---|---|---|---|---|---|
| user_time | train | 13,883,097 | 69.41% | 138,493 | 20,656 | 7,087,069 |
| | val | 2,062,715 | 10.31% | 138,493 | 18,513 | 983,364 |
| | test | 4,054,451 | 20.27% | 138,493 | 25,139 | 1,924,977 |
| user_time_10pct | train | 1,382,688 | 69.42% | 13,671 | 13,616 | 705,219 |
| | val | 205,444 | 10.31% | 13,671 | 10,924 | 97,529 |
| | test | 403,760 | 20.27% | 13,671 | 14,654 | 190,945 |
| global_time | train | 14,000,184 | 70.0% | 100,793 | 9,696 | 7,047,751 |
| | val | 2,000,026 | 10.0% | 16,322 | 12,126 | 949,046 |
| | test | 4,000,053 | 20.0% | 31,670 | 26,349 | 1,998,613 |
| global_time_10pct | train | 1,400,308 | 70.3% | 9,936 | 9,127 | 704,545 |
| | val | 193,560 | 9.72% | 1,605 | 8,229 | 93,261 |
| | test | 398,024 | 19.98% | 3,168 | 14,557 | 195,887 |

Who can be evaluated at each stage (a user needs at least one liked target movie and at least one
training rating):

| Split | Stage | Evaluable users | Target users without history | Target likes on movies unseen in training |
|---|---|---|---|---|
| user_time | val | 130,411 | 0 | 453 of 983,364 |
| user_time | test | 135,599 | 0 | 1,782 of 1,924,977 |
| user_time_10pct | val | 12,905 | 0 | 284 of 97,529 |
| user_time_10pct | test | 13,385 | 0 | 985 of 190,945 |
| global_time | val | 4,538 | 11,673 | 93,714 of 949,046 |
| global_time | test | 5,542 | 26,027 | 300,783 of 1,998,613 |
| global_time_10pct | val | 447 | 1,146 | 9,525 of 93,261 |
| global_time_10pct | test | 568 | 2,589 | 31,287 of 195,887 |

## 5. What `global_time` can and cannot evaluate

`global_time` is the realistic setting: the model is trained on the past and judged on the future
for everyone at once. But it can judge only part of that future:

- At the test stage only **5,542 users** are evaluated, against 135,599 on `user_time`.
- **26,027 users** who rate in the test period have no training history. They are new users, and no
  method here can recommend for them, so they are left out of the evaluation.
- **300,783 of the 1,998,613 test likes** are on movies that do not appear in training (new movies).
  No method can recommend a movie it has never seen, so these likes can never be hits. This caps
  recall on `global_time`.

So `global_time` measures how well a method serves returning users on movies it already knows. It does
not measure cold start, for new users or new movies. Its intervals are also much wider than on
`user_time`, because it evaluates far fewer users.
