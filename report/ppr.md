# Monte Carlo Personalized PageRank (PPR)

## 1. Method

PPR scores movies for a user by random walks on a bipartite graph. Users and movies are the nodes, and
an edge joins a user to every movie they liked (rating ≥ 4.0) in the training part. A movie that is
visited often by walks starting at the user is close to the user's taste.

The implementation (`src/methods/ppr.py`) estimates these scores by simulation:

1. **Walks.** For a user u, `n_walkers` walkers start at u. At each of `n_steps` (20) steps a walker
   moves from its user to a random movie that user liked (this visit is counted), then to a random
   user who liked that movie. Then, with probability `alpha` (the restart), it jumps back to u.
   The visit counts over movies estimate u's PPR scores.
2. **Noise filter.** Movies visited fewer than `min_visits` times are dropped, because a count of one
   or two is mostly noise.
3. **Popularity penalty.** With `beta` > 0, each count is divided by (likes of the movie)^beta, which
   lowers the weight of very popular movies (as in RP3beta).
4. **Top 10.** The 10 highest-scoring movies the user has not rated are recommended (ties go to the
   lower movie id). Users with no like cannot start a walk, and they and any short list are filled
   with the most popular unseen movies.

On Spark, the training data is turned into compressed numpy arrays (user → liked movies, movie →
users who liked it) by `src/methods/_shared.py` and broadcast to every worker. Users are processed in
fixed batches of 256, and the walks of a whole batch are simulated together with numpy. Each batch
has its own random generator, seeded with (612, first user of the batch). Because the batches are
fixed, the results do not depend on the number of cores.

## 2. Parameters and tuning

Tuning used only the validation stage of the 10% pilot (`user_time_10pct`, 12,905 evaluated users).
Unlisted parameters keep their defaults: alpha 0.3, n_walkers 100, n_steps 20, beta 0, min_visits 1.
Sorted by NDCG@10. All runs were made on the group's tuning machine.

| Tag | Parameters | P@10 | R@10 | NDCG@10 | Coverage | Recommend (s) |
|---|---|---|---|---|---|---|
| **a0.3_w2500_b0.3_mv10 (chosen)** | **alpha 0.3, 2500 walkers, beta 0.3, min_visits 10** | **0.0462** | **0.0963** | **0.0800** | **0.0462** | **120.08** |
| a0.5_w1000 | alpha 0.5, 1000 walkers | 0.0453 | 0.0929 | 0.0781 | 0.0214 | 91.58 |
| a0.3_w1000_b0.3_mv5 | alpha 0.3, 1000 walkers, beta 0.3, min_visits 5 | 0.0455 | 0.0933 | 0.0780 | 0.0607 | 137.98 |
| a0.3_w1000_b0.5_mv10 | alpha 0.3, 1000 walkers, beta 0.5, min_visits 10 | 0.0454 | 0.0959 | 0.0776 | 0.1427 | 153.91 |
| a0.3_w2500 | alpha 0.3, 2500 walkers | 0.0440 | 0.0896 | 0.0763 | 0.0142 | 158.4 |
| a0.3_w1000 | alpha 0.3, 1000 walkers | 0.0439 | 0.0882 | 0.0757 | 0.0163 | 98.21 |
| a0.3_w250 | alpha 0.3, 250 walkers | 0.0406 | 0.0826 | 0.0689 | 0.0291 | 69.24 |
| a0.3 | alpha 0.3 | 0.0356 | 0.0731 | 0.0608 | 0.0499 | 65.12 |
| default | (defaults) | 0.0356 | 0.0731 | 0.0608 | 0.0499 | 67.2 |
| a0.5 | alpha 0.5 | 0.0348 | 0.0739 | 0.0606 | 0.0689 | 65.4 |
| a0.1 | alpha 0.1 | 0.0340 | 0.0670 | 0.0581 | 0.0412 | 75.96 |
| a0.3_b0.3 | alpha 0.3, beta 0.3 | 0.0113 | 0.0309 | 0.0226 | 0.7236 | 76.45 |
| a0.5_b0.5 | alpha 0.5, beta 0.5 | 0.0002 | 0.0007 | 0.0004 | 0.5874 | 74.75 |
| a0.3_b0.5 | alpha 0.3, beta 0.5 | 0.0001 | 0.0004 | 0.0002 | 0.5507 | 66.09 |
| a0.3_b0.7 | alpha 0.3, beta 0.7 | 0.0001 | 0.0002 | 0.0002 | 0.4682 | 66.51 |

The chosen configuration is fixed in `tools/chosen.json`.

<!-- sections 3, 4 and 6 follow after the test runs -->

## 5. Findings from tuning

- **More walkers help, with diminishing returns.** At alpha 0.3 with no penalty, NDCG@10 was 0.0608
  with 100 walkers, 0.0689 with 250, 0.0757 with 1000 and 0.0763 with 2500. Most of the gain comes by
  1000 walkers; going to 2500 moved NDCG@10 only from 0.0757 to 0.0763.
- **The popularity penalty `beta` works only together with `min_visits`.** Without a visit filter,
  beta collapses NDCG@10: 0.0226 with beta 0.3, and 0.0002 with beta 0.5 and 0.7 (alpha 0.3, 100
  walkers). Rarely visited movies have tiny like counts, so the penalty gives them the largest boost
  and they crowd out the rest; coverage jumps to 0.7236 with beta 0.3. With min_visits the penalty
  helps instead: with 2500 walkers, beta 0.3 and min_visits 10 gave the best result, 0.0800 against
  0.0763 without the penalty, and coverage rose from 0.0142 to 0.0462.
- **The restart probability `alpha` mattered little** at 100 walkers: 0.0581 (alpha 0.1), 0.0608 (0.3)
  and 0.0606 (0.5).
