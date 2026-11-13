# Paired comparisons between methods

Each row is a paired bootstrap (`tools/compare.py`): the per-user metrics of two test runs are joined on
the user id, users are resampled 1,000 times with the same resample for both methods (seed 612), and the
table gives the mean difference A − B with its 95% interval. P(A ≤ B) is the share of resamples where A
was not better than B. Both runs always evaluate exactly the same users. Logs: `logs/compare/`.

Each method is compared with exact kNN (the reference that LSH approximates), and the best method on both
splits, ALS, is compared with every other method.

## 1. user_time (135,599 users)

| A − B | ΔP@10 [95% CI] | ΔR@10 [95% CI] | ΔNDCG@10 [95% CI] | P(A ≤ B), NDCG |
|---|---|---|---|---|
| LSH − kNN | −0.0001 [−0.0002, +0.0000] | −0.0002 [−0.0005, +0.0000] | −0.0001 [−0.0003, +0.0001] | 0.918 |
| ALS − kNN | +0.0086 [+0.0081, +0.0091] | +0.0076 [+0.0068, +0.0084] | +0.0107 [+0.0100, +0.0114] | 0.000 |
| PPR − kNN | −0.0285 [−0.0291, −0.0279] | −0.0319 [−0.0328, −0.0312] | −0.0382 [−0.0390, −0.0375] | 1.000 |
| ALS − LSH | +0.0088 [+0.0083, +0.0092] | +0.0078 [+0.0070, +0.0086] | +0.0108 [+0.0101, +0.0115] | 0.000 |
| ALS − PPR | +0.0371 [+0.0365, +0.0378] | +0.0395 [+0.0386, +0.0405] | +0.0489 [+0.0480, +0.0498] | 0.000 |
| ALS − popularity | +0.0502 [+0.0495, +0.0509] | +0.0596 [+0.0586, +0.0607] | +0.0668 [+0.0658, +0.0678] | 0.000 |

## 2. global_time (5,542 users)

| A − B | ΔP@10 [95% CI] | ΔR@10 [95% CI] | ΔNDCG@10 [95% CI] | P(A ≤ B), NDCG |
|---|---|---|---|---|
| LSH − kNN | +0.0003 [−0.0007, +0.0012] | +0.0002 [−0.0003, +0.0007] | −0.0005 [−0.0014, +0.0004] | 0.854 |
| ALS − kNN | +0.0096 [+0.0064, +0.0131] | +0.0015 [−0.0006, +0.0038] | +0.0091 [+0.0055, +0.0128] | 0.000 |
| PPR − kNN | −0.0483 [−0.0515, −0.0450] | −0.0133 [−0.0151, −0.0116] | −0.0540 [−0.0578, −0.0504] | 1.000 |
| ALS − LSH | +0.0093 [+0.0059, +0.0126] | +0.0013 [−0.0008, +0.0036] | +0.0096 [+0.0061, +0.0133] | 0.000 |
| ALS − PPR | +0.0579 [+0.0534, +0.0625] | +0.0149 [+0.0123, +0.0175] | +0.0632 [+0.0582, +0.0681] | 0.000 |
| ALS − popularity | +0.0732 [+0.0684, +0.0780] | +0.0181 [+0.0155, +0.0209] | +0.0783 [+0.0729, +0.0836] | 0.000 |

## 3. What the comparisons show

- **The ranking is the same on both splits:** ALS > exact kNN ≈ LSH > PPR > popularity, by NDCG@10.
- **ALS is the best method.** It beats every other method on P@10 and NDCG@10 on both splits, with all
  intervals above zero. On `global_time` its recall advantage over kNN (+0.0015 [−0.0006, +0.0038]) and
  over LSH (+0.0013 [−0.0008, +0.0036]) is not significant.
- **LSH matches exact kNN.** Every LSH − kNN interval contains zero on both splits.
- **PPR is clearly below kNN** on every metric on both splits, though it still beats popularity
  (`report/ppr.md`).
- The intervals on `global_time` are much wider than on `user_time`, because it evaluates 5,542 users
  against 135,599. Differences should only be compared within one split: the two splits evaluate
  different users with different targets (`report/data.md`).
