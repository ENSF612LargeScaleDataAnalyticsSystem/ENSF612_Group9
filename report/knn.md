# Exact user-kNN

## 1. Method

Exact user-kNN recommends to each user the movies liked by the users whose taste is closest to theirs.
It is the reference that MinHash-LSH approximates, so it uses the same scoring.

For a user *u* with the set of liked movies *A_u* (ratings of 4.0 or more in the training part):

1. **Similarity to every other user.** For each movie in *A_u* we look up the users who liked it and count
   how often each user appears. That count is the overlap |A_u ∩ A_v| for every other user *v* at once.
   The similarity is cosine on the liked sets, |A_u ∩ A_v| / sqrt(|A_u| · |A_v|).
2. **Neighbours.** Keep the 100 most similar users. Ties go to the lower user id, so the result does not
   depend on how Spark partitions the data.
3. **Scores.** The score of a movie is the sum of the similarities of the neighbours who liked it. The 10
   best-scoring movies the user has not rated are recommended. A user with no likes, or with fewer than
   10 scored movies, is filled up with the most popular unseen movies.

**Implementation on Spark.** The training ratings are collected once and turned into three compressed
sparse arrays with numpy: user → rated movies, user → liked movies, movie → users who liked it. The arrays
are broadcast to all workers. Users are sorted by id and cut into fixed batches of 256, and the batches
are scored in parallel with `mapPartitions`. Because the batches depend only on the user ids, the
recommendations are identical for any number of cores.

The cost for one user is the total number of likes of the movies that user liked. Summed over users this
is the sum over movies of (number of likes)², so popular movies dominate the run time.

## 2. Parameters and tuning

Tuning used only the validation stage of the 10% pilot sample (`user_time_10pct`, 12,905 evaluated
users). The test part was not read.

| Configuration | P@10 | R@10 | NDCG@10 | Coverage | Recommend s |
|---|---|---|---|---|---|
| **100 neighbours, cosine (chosen)** | 0.0638 | 0.1310 | 0.1081 | 0.0533 | 74 |
| 100 neighbours, Jaccard | 0.0628 | 0.1290 | 0.1065 | 0.0558 | 54 |
| 50 neighbours, Jaccard (default) | 0.0624 | 0.1275 | 0.1057 | 0.0792 | 57 |
| 200 neighbours, Jaccard | 0.0618 | 0.1266 | 0.1050 | 0.0420 | 56 |
| 20 neighbours, Jaccard | 0.0605 | 0.1222 | 0.1021 | 0.1196 | 80 |

Accuracy changes little between 50 and 200 neighbours and is best at 100. Cosine is slightly better than
Jaccard at 100 neighbours. Fewer neighbours give more varied lists: coverage falls from 0.120 with 20
neighbours to 0.042 with 200.

## 3. Test results

One run per split, with the chosen configuration, training on train + validation.

| Split | Method | P@10 [95% CI] | R@10 [95% CI] | NDCG@10 [95% CI] | Coverage | Users | Recommend s |
|---|---|---|---|---|---|---|---|
| user_time | kNN | 0.1034 [0.1027, 0.1041] | 0.1140 [0.1131, 0.1151] | 0.1367 [0.1357, 0.1377] | 0.0875 | 135,599 | 1,634 |
| user_time | popularity | 0.0619 [0.0613, 0.0624] | 0.0619 [0.0612, 0.0626] | 0.0806 [0.0798, 0.0814] | 0.0129 | 135,599 | 38 |
| global_time | kNN | 0.1898 [0.1835, 0.1959] | 0.0503 [0.0481, 0.0528] | 0.2130 [0.2061, 0.2196] | 0.1390 | 5,542 | 1,428 |
| global_time | popularity | 0.1262 [0.1212, 0.1313] | 0.0337 [0.0318, 0.0362] | 0.1438 [0.1383, 0.1494] | 0.0267 | 5,542 | 36 |

All four runs were made on the same machine: 4 CPUs, Windows, Spark 4.0.1 in local mode with all cores
and an 8 GB driver heap.

## 4. Comparison with the baseline

Paired bootstrap over users (1,000 resamples, the same resample for both methods), kNN minus popularity:

| Split | Metric | Difference | 95% interval |
|---|---|---|---|
| user_time | Precision@10 | +0.0415 | [+0.0409, +0.0421] |
| user_time | Recall@10 | +0.0521 | [+0.0511, +0.0530] |
| user_time | NDCG@10 | +0.0561 | [+0.0552, +0.0570] |
| global_time | Precision@10 | +0.0636 | [+0.0599, +0.0672] |
| global_time | Recall@10 | +0.0166 | [+0.0146, +0.0187] |
| global_time | NDCG@10 | +0.0692 | [+0.0648, +0.0733] |

kNN is better than popularity on every metric and both splits, and no interval comes close to zero. On
`user_time`, NDCG@10 rises from 0.0806 to 0.1367. kNN also recommends a much wider part of the catalogue:
coverage is 0.0875 against 0.0129 on `user_time`.

The price is time. Producing the recommendations took 1,634 s on `user_time` against 38 s for popularity
on the same machine, about 43 times longer.

## 5. Limitations

- **Cost grows faster than the data.** The work is the sum over movies of (likes)², so doubling the number
  of users roughly quadruples it. This is the reason for comparing against MinHash-LSH.
- **Memory.** The whole training index is held in the driver and copied to every worker, so the method is
  limited by the memory of one machine.
- **New users and new movies.** A user with no likes only receives popular movies, and a movie nobody in
  training liked can never be recommended.
- **`global_time` evaluates few users.** Only 5,542 users have both training history and a liked movie
  after the cut-off, so its intervals are about seven times wider than on `user_time`.
