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

<!-- sections 3-6 follow after the test runs -->
