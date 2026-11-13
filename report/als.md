# ALS (Spark MLlib, implicit feedback)

## 1. Method

ALS factorises the user–movie matrix into a user factor matrix U and an item factor matrix V of a
chosen rank. The score of a movie for a user is the dot product of their two factor vectors. We use
the implicit-feedback version (Hu, Koren and Volinsky 2008). Every liked pair (rating ≥ 4.0) in the
training part is a positive with value 1, and every other pair is a weak negative. The parameter
`alpha` sets how much more confidence an observed like gets than an unobserved pair. `reg` is the L2
penalty on the factors.

The implementation (`src/methods/als.py`) has two parts:

- **Training on Spark.** `pyspark.ml.recommendation.ALS` is fitted on the liked pairs with
  `implicitPrefs=True`, a fixed seed (612), 10 iterations, and 10 user blocks and 10 item blocks.
  The block counts are fixed so the factors do not change with the number of cores.
- **Exact scoring with numpy.** The factors are collected to the driver as numpy arrays and broadcast
  with the training index from `src/methods/_shared.py`. Users are scored in fixed batches of 256.
  Each batch computes U·Vᵀ for all movies, removes every movie the user rated in training, and keeps
  the 10 best by `top_k` (ties go to the lower movie id). Only movies with a factor are candidates.
  A user with no factor (no like in training) gets the most popular unseen movies, as does any list
  with fewer than 10 candidates. MLlib's `recommendForAllUsers` is not used, because it cannot leave
  out seen movies.

## 2. Parameters and tuning

Tuning used only the validation stage of the 10% pilot (`user_time_10pct`, 12,905 evaluated users).
Unlisted parameters keep their defaults: rank 64, alpha 10, reg 0.05, 10 iterations, likes as the
signal. Sorted by NDCG@10. Recommend times were measured on the group's tuning machine.

| Tag | Parameters | P@10 | R@10 | NDCG@10 | Coverage | Recommend (s) |
|---|---|---|---|---|---|---|
| **r128_a1_reg0.1 (chosen)** | **rank 128, alpha 1, reg 0.1** | **0.0688** | **0.1412** | **0.1170** | **0.1095** | **383.89** |
| r64_a0.3 | rank 64, alpha 0.3 | 0.0668 | 0.1346 | 0.1128 | 0.0720 | 109.03 |
| r64_a1 | rank 64, alpha 1 | 0.0667 | 0.1355 | 0.1127 | 0.0783 | 123.38 |
| r64_a3 | rank 64, alpha 3 | 0.0657 | 0.1356 | 0.1107 | 0.0936 | 131.68 |
| r128_a1 | rank 128, alpha 1 | 0.0643 | 0.1306 | 0.1095 | 0.0997 | 279.28 |
| r128_a1_it15 | rank 128, alpha 1, 15 iterations | 0.0639 | 0.1293 | 0.1091 | 0.1005 | 393.65 |
| r128 | rank 128 | 0.0604 | 0.1281 | 0.1030 | 0.1361 | 282.36 |
| r64_reg0.1 | rank 64, reg 0.1 | 0.0618 | 0.1283 | 0.1025 | 0.1236 | 124.65 |
| default | (defaults) | 0.0610 | 0.1275 | 0.1019 | 0.1186 | 115.95 |
| r64 | rank 64 | 0.0610 | 0.1275 | 0.1019 | 0.1186 | 116.22 |
| r64_reg0.01 | rank 64, reg 0.01 | 0.0609 | 0.1267 | 0.1015 | 0.1144 | 122.41 |
| r32 | rank 32 | 0.0614 | 0.1240 | 0.1008 | 0.0994 | 78.68 |
| r64_a40 | rank 64, alpha 40 | 0.0521 | 0.1129 | 0.0870 | 0.1543 | 122.21 |
| r64_rating | rank 64, ratings as confidence | 0.0399 | 0.0902 | 0.0659 | 0.2062 | 137.75 |

The chosen configuration (rank 128, alpha 1.0, reg 0.1) is fixed in `tools/chosen.json`. Before the
test runs, this machine re-ran it on the pilot and reproduced the group's NDCG@10 of 0.117001 exactly.

## 3. Test results

Train + validation → test, one run per split, with the chosen parameters.

| Method | Split | P@10 [95% CI] | R@10 [95% CI] | NDCG@10 [95% CI] | Coverage | Evaluated users | Recommend (s) |
|---|---|---|---|---|---|---|---|
| ALS | user_time | 0.1120 [0.1113, 0.1128] | 0.1216 [0.1206, 0.1226] | 0.1474 [0.1464, 0.1484] | 0.0854 | 135,599 | 159.06 |
| Popularity | user_time | 0.0619 [0.0613, 0.0624] | 0.0619 [0.0612, 0.0626] | 0.0806 [0.0798, 0.0814] | 0.0129 | 135,599 | 38.45 |
| ALS | global_time | 0.1994 [0.1929, 0.2054] | 0.0519 [0.0497, 0.0546] | 0.2221 [0.2150, 0.2288] | 0.1613 | 5,542 | 162.62 |
| Popularity | global_time | 0.1262 [0.1212, 0.1313] | 0.0337 [0.0318, 0.0362] | 0.1438 [0.1383, 0.1494] | 0.0267 | 5,542 | 35.53 |

The ALS runs were made on macOS 27.0.1 (arm64), 10 CPUs, Spark 4.0.1, 8g driver memory, all cores.
The popularity rows come from the group's runs on Windows 10 (AMD64), 4 CPUs, 8g driver memory.
Run times from the two machines are not comparable.

## 4. Comparison with the baseline

Paired bootstrap (1,000 resamples of the same users), ALS − popularity:

| Split | Users | ΔP@10 [95% CI] | ΔR@10 [95% CI] | ΔNDCG@10 [95% CI] |
|---|---|---|---|---|
| user_time | 135,599 | +0.0502 [+0.0495, +0.0509] | +0.0596 [+0.0586, +0.0607] | +0.0668 [+0.0658, +0.0678] |
| global_time | 5,542 | +0.0732 [+0.0684, +0.0780] | +0.0181 [+0.0155, +0.0209] | +0.0783 [+0.0729, +0.0836] |

ALS beats the popularity baseline on every metric on both splits. No interval comes near zero, and
the share of resamples where ALS was not better is 0.000 in all six cases. On `user_time`, NDCG@10
rises from 0.0806 to 0.1474. ALS also recommends a much wider part of the catalogue (coverage 0.0854
against 0.0129 on `user_time`).

## 5. Findings from tuning and comparison with kNN

- **The confidence weight `alpha` mattered most.** At rank 64, NDCG@10 was 0.1128 with alpha 0.3,
  0.1127 with alpha 1, 0.1107 with alpha 3, 0.1019 with alpha 10 (the default) and 0.0870 with
  alpha 40. Alpha 0.3 and 1 are practically tied (difference 0.0001); alpha 1 was kept for the larger
  rank. Using the raw ratings as confidence instead of likes was much worse (0.0659), although it
  gave the highest coverage (0.2062).
- **Rank and reg helped only slightly.** With alpha 10, rank 32, 64 and 128 gave
  0.1008, 0.1019 and 0.1030. With alpha 1, going from rank 64 to 128 alone lowered NDCG@10 (0.1127 →
  0.1095). Raising reg from 0.05 to 0.1 at rank 128 then gave the best result, 0.1170. Without a
  rank-64, alpha-1, reg-0.1 run we cannot separate the effect of rank from that of reg. 15 iterations
  instead of 10 did not help (0.1091 vs 0.1095).
- **ALS against exact user-kNN** on `user_time` (test, 135,599 users, paired bootstrap):

| Metric | ALS | kNN | ALS − kNN [95% CI] |
|---|---|---|---|
| P@10 | 0.1120 | 0.1034 | +0.0086 [+0.0081, +0.0091] |
| R@10 | 0.1216 | 0.1140 | +0.0076 [+0.0068, +0.0084] |
| NDCG@10 | 0.1474 | 0.1367 | +0.0107 [+0.0100, +0.0114] |

  ALS is better than exact kNN on all three metrics, by about 0.01 NDCG@10. The intervals exclude
  zero. Coverage is close (ALS 0.0854, kNN 0.0875).

## 6. Limitations

- **Memory and the driver.** Both factor matrices are collected to the driver and broadcast to every
  worker, and each batch builds a dense 256 × (number of movies) score matrix. This fits MovieLens
  20M on one machine but would not scale to a much larger catalogue without approximate scoring.
- **Run time.** Recommend time (training plus scoring) grows with the rank. On the pilot, rank 64 took 116.22 s and rank 128
  took 282.36 s, and the chosen configuration took 383.89 s.
- **What `global_time` can evaluate.** Only users who rated before the cutoff and liked a movie after
  it are evaluated (5,542 users, against 135,599 on `user_time`), so its intervals are much wider.
  Its recall is low (0.0519) while its precision is high (0.1994): with only 10 slots per list,
  recall stays low when a user liked many movies in the test part.
- **Tuning scope.** Parameters were tuned on the `user_time` pilot only and reused for `global_time`.
  The grid was small: one parameter changed at a time from a few starting points.
