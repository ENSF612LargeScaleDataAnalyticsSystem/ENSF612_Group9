# Evaluation protocol

All five methods (popularity, exact user-kNN, MinHash-LSH, Monte Carlo PPR and ALS) go through the
same script, `src/run.py`. It loads one split and stage, calls the method's `recommend` function with
the training data only, writes the top-10 lists, and scores them with `src/evaluate.py`. So every
method is scored by the same code on the same users.

## 1. Stages

Each split (`user_time`, `global_time` and their 10% pilots) has three time-ordered parts: train,
validation and test.

- **Validation stage** (`--stage val`): the method trains on the train part and is scored on the
  validation part. All tuning was done here, on the 10% pilot of `user_time` (`user_time_10pct`).
- **Test stage** (`--stage test`): the method trains on train + validation and is scored on the test
  part. It runs once per method and split, with the parameters fixed in `tools/chosen.json`.
  `tools/member.py test` reads the parameters from that file, accepts none from the command line,
  and refuses to run a split again once it has a `metrics.json`.

## 2. Who is evaluated

A user is evaluated if they have at least one liked item (rating ≥ 4.0) in the target part and at
least one rating in the training part. The relevant items of a user are the items they liked in the
target part. An evaluated user who gets no list from a method still counts, with 0 for every metric,
so a method cannot gain by skipping hard users. Before scoring, the evaluation rejects lists with
duplicate items, duplicate ranks or ranks outside 1..10. It warns if a method recommends a movie the
user already rated in training. Such an item can never be a hit.

## 3. Metrics

All metrics use k = 10 and binary relevance (an item is a hit if the user liked it in the target part).
For one user with list positions r = 1..10 and R liked target items:

- **Precision@10** = hits / 10.
- **Recall@10** = hits / R.
- **NDCG@10** = DCG / IDCG, with DCG = Σ over hits of 1 / log2(r + 1), and IDCG the same sum for the
  first min(R, 10) positions (a perfect list).

Each is averaged over all evaluated users.

- **Coverage** = number of distinct items recommended to anyone / number of distinct items in the
  training part.

## 4. Uncertainty

Each metric has a 95% interval from a percentile bootstrap: 1,000 resamples of the evaluated users
with replacement, with a fixed seed (612). Users are sorted by id first, so every machine draws the
same resamples. Per-user values are saved in `results/<method>/<split>/<stage>/per_user`.

To compare two methods, `tools/compare.py` uses a **paired bootstrap**. It joins the per-user values
of both runs on the user id. It refuses to run if the two runs did not evaluate exactly the same
users. Then it resamples users 1,000 times, using the same resample for both methods. It reports the
mean difference A − B, its 95% interval, and the share of resamples where A is not better than B (a
one-sided p-value). Pairing removes the variation between users that both methods share, so the
interval for a difference is narrower than two separate intervals would suggest.

## 5. Why there is no leakage

- A method receives only the training DataFrame. The method contract forbids reading the
  validation or test files, and `src/run.py` passes nothing else.
- The splits are ordered in time. In `global_time`, two global timestamp cutoffs separate the parts,
  so a method never trains on a rating made after any rating it is scored on. In `user_time`, each
  user's last 20% of ratings are test and the 10% before them are validation. No user's own later
  ratings reach training, but the cutoff differs per user, so training can hold other users'
  ratings made after this user's test ratings. Only `global_time` gives a strict time order across
  users.
- Methods never recommend an item the user already rated in training.
- Tuning used only the validation stage of the 10% pilot. Parameters were fixed in
  `tools/chosen.json` before any test run, and each test split runs only once.
- The popularity and kNN test runs were made on the group's reference machine. Every other member
  runs the test stage on their own machine, and only after two checks. `tools/member.py check`
  confirms that the machine's data profile and split statistics are identical to the group's
  copies, so every machine scores on the same rows. `tools/member.py pilot` then re-runs the chosen
  parameters on the pilot's validation stage and must reproduce the group's metrics (within 0.002
  per metric, with the same evaluated users).

## 6. Reproducibility

Results do not depend on the number of cores or on Spark's partitioning:

- Users are scored in fixed batches of 256 sorted user ids, so batches depend only on the ids.
- Every top-n selection breaks ties by the lower id (`best_idx`), so ties never depend on input order.
- Random parts are seeded (seed 612). PPR seeds its random walks per batch with (seed, first user of
  the batch). LSH draws its hash functions from the seed. ALS uses the seed and fixed numbers of user
  and item blocks, so its factors do not change with the number of cores.
- Samples are chosen by a hash of the user id, not by a random draw, so they are nested
  (10% ⊂ 25% ⊂ 50%) and the same on every machine.
- The bootstrap sorts users by id before resampling.
