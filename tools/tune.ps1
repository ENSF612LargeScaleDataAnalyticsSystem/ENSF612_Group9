# Tuning grids on the pilot validation stage. One run per configuration, logged to logs\tune_<method>_<tag>.log,
# results in results\<method>\user_time_10pct\val__<tag> and one row each in results\summary.csv.
#   .\tools\tune.ps1 knn          # one method
#   .\tools\tune.ps1 knn lsh ppr als
# Bootstrap is skipped (--n-boot 0) to save time; the chosen configuration gets CIs in the final test run.
$grids = @{
    knn = @(
        @('n20',  'n_neighbours=20'), @('n50', 'n_neighbours=50'), @('n100', 'n_neighbours=100'),
        @('n200', 'n_neighbours=200'), @('n100_cos', 'n_neighbours=100', 'similarity=cosine')
    )
    # threshold ~ (1/bands)^(1/r), r = n_hashes/bands: r=4,b=32 -> 0.42; r=2,b=64 -> 0.125; r=1 -> any overlap
    lsh = @(
        @('h128_b64',  'n_hashes=128', 'bands=64'),  @('h256_b128', 'n_hashes=256', 'bands=128'),
        @('h64_b64',   'n_hashes=64',  'bands=64'),  @('h128_b128', 'n_hashes=128', 'bands=128'),
        @('h128_b64_mb5000', 'n_hashes=128', 'bands=64', 'max_bucket=5000')
    )
    ppr = @(
        @('a0.1', 'alpha=0.1'), @('a0.3', 'alpha=0.3'), @('a0.5', 'alpha=0.5'),
        @('a0.3_b0.3', 'alpha=0.3', 'beta=0.3'), @('a0.3_b0.5', 'alpha=0.3', 'beta=0.5'),
        @('a0.3_b0.7', 'alpha=0.3', 'beta=0.7'), @('a0.5_b0.5', 'alpha=0.5', 'beta=0.5'),
        @('a0.3_w250', 'alpha=0.3', 'n_walkers=250')
    )
    # second PPR round: more walkers is nearly free (overhead dominates), and min_visits tames beta's noise
    ppr2 = @(
        @('a0.3_w1000', 'alpha=0.3', 'n_walkers=1000'), @('a0.3_w2500', 'alpha=0.3', 'n_walkers=2500'),
        @('a0.5_w1000', 'alpha=0.5', 'n_walkers=1000'),
        @('a0.3_w1000_b0.3_mv5', 'alpha=0.3', 'n_walkers=1000', 'beta=0.3', 'min_visits=5'),
        @('a0.3_w1000_b0.5_mv10', 'alpha=0.3', 'n_walkers=1000', 'beta=0.5', 'min_visits=10'),
        @('a0.3_w2500_b0.3_mv10', 'alpha=0.3', 'n_walkers=2500', 'beta=0.3', 'min_visits=10')
    )
    # second ALS round: alpha 1 beat 10 and 40 (0.1127 vs 0.1019 / 0.0870), so go lower and combine with
    # the rank / reg values that helped
    als2 = @(
        @('r64_a0.3', 'rank=64', 'alpha=0.3'), @('r64_a3', 'rank=64', 'alpha=3.0'),
        @('r128_a1', 'rank=128', 'alpha=1.0'), @('r128_a1_reg0.1', 'rank=128', 'alpha=1.0', 'reg=0.1'),
        @('r128_a1_it15', 'rank=128', 'alpha=1.0', 'max_iter=15')
    )
    als = @(
        @('r32',  'rank=32'), @('r64', 'rank=64'), @('r128', 'rank=128'),
        @('r64_reg0.01', 'rank=64', 'reg=0.01'), @('r64_reg0.1', 'rank=64', 'reg=0.1'),
        @('r64_a1', 'rank=64', 'alpha=1.0'), @('r64_a40', 'rank=64', 'alpha=40.0'),
        @('r64_rating', 'rank=64', 'signal=rating')
    )
}
$methods = if ($args.Count) { $args } else { 'knn', 'lsh', 'ppr', 'als' }
foreach ($m in $methods) {
    if (-not $grids.ContainsKey($m)) { "unknown grid $m"; continue }
    $method = $m -replace '\d+$', ''          # grid 'ppr2' -> method 'ppr'
    foreach ($cfg in $grids[$m]) {
        $tag = $cfg[0]
        $pargs = @()
        foreach ($p in $cfg[1..($cfg.Count - 1)]) { $pargs += '--param'; $pargs += $p }
        & "$PSScriptRoot\run.ps1" "tune_${m}_$tag" -m src.run --method $method --split user_time_10pct --stage val `
            --n-boot 0 --tag $tag @pargs
        "{0} {1}: exit {2}" -f $m, $tag, $LASTEXITCODE
    }
}
