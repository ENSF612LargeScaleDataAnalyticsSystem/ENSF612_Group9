# Final test runs: once per method and split, with the parameters chosen on validation (tools\chosen.json).
#   .\tools\final.ps1                 # knn lsh ppr als on user_time and global_time
#   .\tools\final.ps1 als             # one method
# Popularity has no parameters and was already run on test.
$chosen = Get-Content "$PSScriptRoot\chosen.json" -Raw | ConvertFrom-Json
$methods = if ($args.Count) { $args } else { 'knn', 'lsh', 'ppr', 'als' }
foreach ($m in $methods) {
    $pargs = @()
    if ($chosen.$m) { foreach ($p in $chosen.$m) { $pargs += '--param'; $pargs += $p } }
    foreach ($split in 'user_time', 'global_time') {
        & "$PSScriptRoot\run.ps1" "final_${m}_$split" -m src.run --method $m --split $split --stage test @pargs
        "{0} {1}: exit {2}" -f $m, $split, $LASTEXITCODE
    }
}
