# MinHash-LSH user-kNN

## 1. Method

LSH replaces the exact neighbour search of user-kNN with an approximate one. Everything after the
neighbour search is the same as in exact kNN (`src/methods/knn.py`), so any difference between the
two methods comes from the neighbours alone.

The implementation (`src/methods/lsh.py`) works on each user's set of liked movies (rating ≥ 4.0)
in the training part:

1. **MinHash signatures.** Each user's liked set gets a signature of `n_hashes` values. Value j is the
   minimum of h_j(x) = (a_j·x + b_j) mod (2^31 − 1) over the liked movies x. The a_j and b_j come from
   a generator seeded with 612. Signatures are computed in fixed batches of 256 users on Spark
   workers, from the training index broadcast by `src/methods/_shared.py`.
2. **Banding.** The signature is cut into `bands` bands of r = n_hashes / bands values. Each band is
   hashed into one key. Spark groups users by (band, key), and users that share a bucket in at least
   one band become candidates for each other. A pair with Jaccard similarity s is found with
   probability 1 − (1 − s^r)^bands.
3. **Big buckets are skipped.** Buckets with more than `max_bucket` users (default 2,000) are left
   out. They come from very popular movies and would bring back the quadratic cost LSH is meant to
   avoid.
4. **Exact verification.** For every candidate the exact similarity (cosine or Jaccard on liked sets)
   is computed, and the `n_neighbours` best are kept (ties go to the lower user id).
5. **Scoring as in kNN.** A movie's score is the sum of the similarities of the neighbours who liked
   it. The 10 best movies the user has not rated are recommended. Users with no like or too few
   scored movies are filled with the most popular unseen movies.

## 2. Parameters and tuning

Tuning used only the validation stage of the 10% pilot (`user_time_10pct`, 12,905 evaluated users).
Unlisted parameters keep their defaults: n_hashes 128, bands 32, n_neighbours 50, max_bucket 2,000,
Jaccard similarity. Sorted by NDCG@10. All runs were made on the group's tuning machine.

| Tag | Parameters | P@10 | R@10 | NDCG@10 | Coverage | Recommend (s) |
|---|---|---|---|---|---|---|
| **chosen** | **n_hashes 256, bands 128, cosine, n_neighbours 100** | **0.0633** | **0.1295** | **0.1068** | **0.0773** | **152.6** |
| h64_b64 | n_hashes 64, bands 64 | 0.0624 | 0.1277 | 0.1058 | 0.0800 | 265.56 |
| h128_b128 | n_hashes 128, bands 128 | 0.0624 | 0.1276 | 0.1057 | 0.0800 | 309.1 |
| h256_b128 | n_hashes 256, bands 128 | 0.0623 | 0.1265 | 0.1049 | 0.0908 | 156.24 |
| h128_b64 | n_hashes 128, bands 64 | 0.0614 | 0.1236 | 0.1027 | 0.1028 | 128.35 |
| h128_b64_mb5000 | n_hashes 128, bands 64, max_bucket 5,000 | 0.0614 | 0.1236 | 0.1027 | 0.1028 | 116.91 |
| default | (defaults) | 0.0463 | 0.0994 | 0.0801 | 0.1537 | 112.49 |

The chosen configuration is fixed in `tools/chosen.json`. Before the test runs, this machine re-ran
it on the pilot and reproduced the group's NDCG@10 of 0.106837 exactly.

## 3. Test results

Train + validation → test, one run per split, with the chosen parameters.

| Method | Split | P@10 [95% CI] | R@10 [95% CI] | NDCG@10 [95% CI] | Coverage | Evaluated users | Recommend (s) |
|---|---|---|---|---|---|---|---|
| LSH | user_time | 0.1033 [0.1025, 0.1040] | 0.1138 [0.1128, 0.1148] | 0.1366 [0.1355, 0.1376] | 0.0879 | 135,599 | 522.59 |
| Popularity | user_time | 0.0619 [0.0613, 0.0624] | 0.0619 [0.0612, 0.0626] | 0.0806 [0.0798, 0.0814] | 0.0129 | 135,599 | 38.45 |
| LSH | global_time | 0.1901 [0.1835, 0.1961] | 0.0506 [0.0483, 0.0532] | 0.2125 [0.2054, 0.2190] | 0.1413 | 5,542 | 496.56 |
| Popularity | global_time | 0.1262 [0.1212, 0.1313] | 0.0337 [0.0318, 0.0362] | 0.1438 [0.1383, 0.1494] | 0.0267 | 5,542 | 35.53 |

The LSH runs were made on macOS 27.0.1 (arm64), 10 CPUs, Spark 4.0.1, 14g driver memory, all cores.
The popularity rows come from the group's runs on Windows 10 (AMD64), 4 CPUs, 8g driver memory.
Run times from the two machines are not comparable.

During the test runs LSH reported 12,493.0 candidates per user on average on `user_time` (360 users
without candidates; 206 buckets with more than 2,000 users skipped), and 13,221.8 on `global_time`
(142 users without candidates; 165 buckets skipped).

## 4. Comparison with the baseline

Paired bootstrap (1,000 resamples of the same users), LSH − popularity:

| Split | Users | ΔP@10 [95% CI] | ΔR@10 [95% CI] | ΔNDCG@10 [95% CI] |
|---|---|---|---|---|
| user_time | 135,599 | +0.0414 [+0.0408, +0.0421] | +0.0519 [+0.0509, +0.0527] | +0.0560 [+0.0551, +0.0568] |
| global_time | 5,542 | +0.0639 [+0.0603, +0.0675] | +0.0168 [+0.0147, +0.0190] | +0.0687 [+0.0643, +0.0728] |

LSH beats the popularity baseline on every metric on both splits. No interval comes near zero, and
the share of resamples where LSH was not better is 0.000 in all six cases. On `user_time`, NDCG@10
rises from 0.0806 to 0.1366, and coverage rises from 0.0129 to 0.0879.

## 5. Approximation quality

**Neighbour quality.** `tools/lsh_recall.py` compares the LSH top-100 neighbours with the exact
top-100 for a sample of users, on the training part of the validation stage (never the test part).

| Data | Users compared | Neighbour recall@100 | Similarity ratio | Candidates per user | Candidate step (s) |
|---|---|---|---|---|---|
| 10% pilot (group machine, from `tasks/lsh.md`) | | 0.8247 | 0.9464 | 1,885.7 | 78 |
| Full `user_time` (this machine) | 1,993 | 0.9164 | 0.9898 | 11,665.3 (median 11,371) | 108.8 |

Both rows use n_hashes 256, bands 128 (2 values per band), cosine similarity and max_bucket 2,000.
The candidate step time covers all users; the other columns are averages over the sampled users.

**Recommendation quality.** Paired bootstrap on `user_time` (test, 135,599 users), LSH − exact kNN:

| Metric | LSH | kNN | LSH − kNN [95% CI] | P(LSH ≤ kNN) |
|---|---|---|---|---|
| P@10 | 0.1033 | 0.1034 | −0.0001 [−0.0002, +0.0000] | 0.932 |
| R@10 | 0.1138 | 0.1140 | −0.0002 [−0.0005, +0.0000] | 0.942 |
| NDCG@10 | 0.1366 | 0.1367 | −0.0001 [−0.0003, +0.0001] | 0.918 |

What is given up: on the full data LSH finds 0.9164 of the exact top-100 neighbours, and the
neighbours it keeps have 0.9898 of the exact similarity mass. What is kept: the recommendations are
almost the same as exact kNN. The NDCG@10 difference is −0.0001 and its interval includes zero.
What is gained: the exact similarity is computed only for the candidates (11,665.3 per sampled user
on the full data) instead of every user who shares a liked movie. The kNN test run was made on the
group's Windows machine (4 CPUs) and the LSH run on this Mac (10 CPUs), so their recommend times are
not comparable; the time comparison is in the scalability section.

## 6. Limitations

- **Memory.** The first `user_time` test run on this machine stopped with a Java heap
  `OutOfMemoryError` at the default 8g driver memory, before anything was scored. The rerun with 14g
  finished. The heap ran out while Spark was building and caching the per-user candidate lists,
  with 10 tasks running at once.
- **Candidates.** With 256 hashes in 128 bands (2 values per band) the banding is loose: on the full
  `user_time` data each user still got 12,493.0 candidates on average, and every candidate's exact
  similarity is computed. Very popular buckets (more than 2,000 users) are skipped, so a few true neighbours that
  only share such a bucket can be missed.
- **What `global_time` can evaluate.** Only users who rated before the cutoff and liked a movie after
  it are evaluated (5,542 users, against 135,599 on `user_time`), so its intervals are much wider.
- **Tuning scope.** Parameters were tuned on the `user_time` pilot only and reused for `global_time`,
  and the LSH − kNN comparison covers only `user_time` (the integrator adds `global_time`).
