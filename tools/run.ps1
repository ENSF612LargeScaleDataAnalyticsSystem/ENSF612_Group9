# Run one pipeline command with Java/Hadoop set and the output logged.
#   .\tools\run.ps1 <logname> -m src.run --method popularity --split user_time --stage test
# Output goes to logs\<logname>.log; the exit code is appended as the last line.
# Uses $args (no param block) so flags such as -m or -p are passed to Python untouched.
if ($args.Count -lt 2) { Write-Error "usage: .\tools\run.ps1 <logname> <python args...>"; exit 2 }
$Log = $args[0]
$PyArgs = $args[1..($args.Count - 1)]

$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
if (-not $env:JAVA_HOME) { $env:JAVA_HOME = [Environment]::GetEnvironmentVariable('JAVA_HOME', 'User') }
if (-not $env:HADOOP_HOME) { $env:HADOOP_HOME = [Environment]::GetEnvironmentVariable('HADOOP_HOME', 'User') }
$env:PATH = "$env:JAVA_HOME\bin;$env:HADOOP_HOME\bin;$env:PATH"

New-Item -ItemType Directory -Force logs | Out-Null
$logPath = "logs\$Log.log"
& .\.venv\Scripts\python.exe @PyArgs *> $logPath
$code = $LASTEXITCODE
"exit: $code" | Out-File -Append -Encoding utf8 $logPath
exit $code
