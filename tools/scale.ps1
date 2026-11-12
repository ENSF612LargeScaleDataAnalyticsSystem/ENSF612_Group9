# Scalability runs (RQ3) on the user_time validation stage (the test part is never touched here).
#   .\tools\scale.ps1                 # all methods
#   .\tools\scale.ps1 knn lsh         # some methods
# 1. data size: user samples 10 / 25 / 50 / 100% with all 4 cores
# 2. cores:     1 / 2 cores on the 25% sample (4 cores comes from step 1)
# Every run uses the same --driver-memory and the tuned parameters in tools\chosen.json:
#   { "knn": ["n_neighbours=100"], "lsh": ["n_hashes=128", "bands=64"], ... }
# Results: results\<method>\user_time\val__scale_c<cores>_s<pct>, one row each in results\summary.csv.
$root = Split-Path $PSScriptRoot -Parent
$chosen = Get-Content "$PSScriptRoot\chosen.json" -Raw | ConvertFrom-Json
$methods = if ($args.Count) { $args } else { 'popularity', 'knn', 'lsh', 'ppr', 'als' }
$mem = '8g'

$runs = @()
foreach ($s in 0.1, 0.25, 0.5, 1.0) { $runs += , @('4', $s) }
foreach ($c in '1', '2') { $runs += , @($c, 0.25) }

foreach ($m in $methods) {
    $pargs = @()
    if ($chosen.$m) { foreach ($p in $chosen.$m) { $pargs += '--param'; $pargs += $p } }
    foreach ($r in $runs) {
        $cores, $sample = $r[0], $r[1]
        $tag = "scale_c$cores"
        $log = "scale_${m}_c${cores}_s$([int]($sample * 100))"
        & "$PSScriptRoot\run.ps1" $log -m src.run --method $m --split user_time --stage val --n-boot 0 `
            --cores $cores --driver-memory $mem --sample $sample --tag $tag @pargs
        "{0} cores={1} sample={2}: exit {3}" -f $m, $cores, $sample, $LASTEXITCODE
    }
}
