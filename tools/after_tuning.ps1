# Everything after tuning, unattended. Progress lines go to logs\after_tuning.log.
#   .\tools\after_tuning.ps1 [-NoWait]
# 1. waits until the tuning grid (logs\tune_all.log) has finished all 26 configurations
# 1b. extra rounds: tune.ps1 ppr2 (more walkers, beta with min_visits) and als2 (lower alpha, combinations)
# 2. tools\pick.py -> tools\chosen.json
# 3. LSH on the pilot with the chosen settings (kNN's similarity / n_neighbours), for the val table
# 4. tools\lsh_recall.py for the chosen banding
# 5. tools\final.ps1   (test stage, user_time + global_time, all four methods)
# 6. tools\scale.ps1   (RQ3 sweep)
# 7. tools\summarize.py -> reports\results.md
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$log = "logs\after_tuning.log"
function Note($msg) { "{0:HH:mm:ss} {1}" -f (Get-Date), $msg | Tee-Object -Append -FilePath $log }

if ($args -notcontains '-NoWait') {
    Note "waiting for tuning to finish"
    while (((Select-String -Path logs\tune_all.log -Pattern 'exit' -ErrorAction SilentlyContinue) | Measure-Object).Count -lt 26) {
        Start-Sleep -Seconds 30
    }
}
Note "tuning finished"

Note "extra rounds (ppr2, als2) start"
& "$PSScriptRoot\tune.ps1" ppr2 als2 *>> logs\tune_extra.log
Note "extra rounds done"

& .\.venv\Scripts\python.exe tools\pick.py *>> $log
$chosen = Get-Content tools\chosen.json -Raw | ConvertFrom-Json

$pargs = @(); foreach ($p in $chosen.lsh) { $pargs += '--param'; $pargs += $p }
& "$PSScriptRoot\run.ps1" tune_lsh_chosen -m src.run --method lsh --split user_time_10pct --stage val --n-boot 0 --tag chosen @pargs
Note "lsh chosen on pilot val: exit $LASTEXITCODE"

$lp = @{}; foreach ($p in $chosen.lsh) { $k, $v = $p -split '=', 2; $lp[$k] = $v }
$rargs = @('--n-hashes', $lp['n_hashes'], '--bands', $lp['bands'])
if ($lp['max_bucket']) { $rargs += @('--max-bucket', $lp['max_bucket']) }
if ($lp['n_neighbours']) { $rargs += @('--n-neighbours', $lp['n_neighbours']) }
if ($lp['similarity']) { $rargs += @('--similarity', $lp['similarity']) }
& "$PSScriptRoot\run.ps1" lsh_recall_pilot tools/lsh_recall.py --split user_time_10pct @rargs
Note "lsh_recall on pilot: exit $LASTEXITCODE"

Note "final test runs start"
& "$PSScriptRoot\final.ps1" *>> $log
Note "final test runs done"

& .\.venv\Scripts\python.exe tools\summarize.py *>> $log
Note "scalability sweep start"
& "$PSScriptRoot\scale.ps1" *>> $log
Note "scalability sweep done"

& .\.venv\Scripts\python.exe tools\summarize.py *>> $log
Note "all done"
