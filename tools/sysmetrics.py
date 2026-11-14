"""Shuffle volume and peak memory per run (RQ2/RQ3): the system metrics the scalability runs lack.

    python tools/sysmetrics.py --driver-memory 14g            # all five methods, 10/25/50/100% users
    python tools/sysmetrics.py knn lsh --samples 0.25 1.0     # some methods / sizes
    python tools/sysmetrics.py --report                       # only rebuild the tables from the saved runs

Each run is the same as a scalability run (user_time validation stage, chosen parameters from
tools/chosen.json, 4 cores, no bootstrap), started as `python -m src.run` in its own process. While it
runs, this script
  * reads the Spark monitoring REST API of that process (http://localhost:4040.., /api/v1) and keeps
    the stage list: shuffle read/write bytes, spilled bytes and input bytes, summed over all stages;
  * samples the resident memory of the whole process tree (Python driver, Spark JVM, Python workers)
    every 0.5 s with `ps` and keeps the peak; also the JVM heap peak that Spark reports per executor.
The recommend phase is separated from the evaluation by time: stages that finished before
(finish time of the run - seconds_evaluate) belong to `recommend`. Evaluation is the same code for
every method, so the recommend columns are the ones to compare.

Wall-clock times are measured again here but are NOT the RQ3 timings (polling adds a little load);
use the scale runs for time. Run on the same machine as the scale runs, with nothing else running.

Writes results/<method>/user_time/val__sys_c4[_s<pct>]/{metrics.json, sysmetrics.json} and the tables
reports/sysmetrics.md and reports/sysmetrics.json.
"""
import argparse
import json
import platform
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from member import ROOT, chosen_params, spark_env  # noqa: E402

METHODS = ("popularity", "knn", "lsh", "ppr", "als")
PORTS = range(4040, 4050)


def folder(method, sample):
    tag = "sys_c4" + (f"_s{sample * 100:g}" if sample < 1 else "")
    return ROOT / "results" / method / "user_time" / f"val__{tag}", tag


def api(port, path):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/api/v1{path}", timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def tree_rss_bytes(root_pid):
    """Resident memory of root_pid and all its descendants, from `ps` (macOS and Linux)."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss=,comm="], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return None
    procs = {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3 and parts[0].isdigit():
            procs[int(parts[0])] = (int(parts[1]), int(parts[2]) * 1024, parts[3] if len(parts) > 3 else "")
    keep, frontier = {root_pid}, [root_pid]
    while frontier:
        p = frontier.pop()
        for pid, (ppid, _, _) in procs.items():
            if ppid == p and pid not in keep:
                keep.add(pid)
                frontier.append(pid)
    total = jvm = 0
    for pid in keep:
        if pid in procs:
            total += procs[pid][1]
            if "java" in procs[pid][2].lower():
                jvm += procs[pid][1]
    return total, jvm


class Watcher(threading.Thread):
    """Polls the Spark REST API and the memory of the run's process tree until stopped."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.stop_flag = pid, threading.Event()
        self.port = self.app = None
        self.stages, self.heap_peak = {}, 0
        self.rss_peak = self.jvm_peak = 0

    def run(self):
        while not self.stop_flag.is_set():
            m = tree_rss_bytes(self.pid)
            if m:
                self.rss_peak, self.jvm_peak = max(self.rss_peak, m[0]), max(self.jvm_peak, m[1])
            if self.app is None:
                for port in PORTS:
                    apps = api(port, "/applications")
                    if apps:
                        self.port, self.app = port, apps[0]["id"]
                        break
            else:
                for s in api(self.port, f"/applications/{self.app}/stages?status=complete") or []:
                    self.stages[(s["stageId"], s.get("attemptId", 0))] = s
                for e in api(self.port, f"/applications/{self.app}/executors") or []:
                    h = (e.get("peakMemoryMetrics") or {}).get("JVMHeapMemory")
                    if h:
                        self.heap_peak = max(self.heap_peak, h)
            self.stop_flag.wait(0.5)


def utc(s):
    """Spark REST timestamps look like 2026-11-14T02:03:04.567GMT."""
    return datetime.strptime(s.replace("GMT", ""), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def sums(stages):
    keys = {"shuffleWriteBytes": "shuffle_write_bytes", "shuffleReadBytes": "shuffle_read_bytes",
            "memoryBytesSpilled": "memory_spilled_bytes", "diskBytesSpilled": "disk_spilled_bytes",
            "inputBytes": "input_bytes"}
    out = {v: int(sum(s.get(k, 0) or 0 for s in stages)) for k, v in keys.items()}
    out["stages"] = len(stages)
    return out


def run_one(method, sample, driver_memory):
    out, tag = folder(method, sample)
    flags, _ = chosen_params(method)
    cmd = [sys.executable, "-m", "src.run", "--method", method, "--split", "user_time", "--stage", "val",
           "--cores", "4", "--driver-memory", driver_memory, "--n-boot", "0", "--tag", "sys_c4", *flags]
    if sample < 1:
        cmd += ["--sample", str(sample)]
    log = ROOT / "logs" / f"sys_{method}_{tag}.log"
    log.parent.mkdir(exist_ok=True)
    print(f"run  {method:<11} sample {sample * 100:>3g}%  (log: {log.relative_to(ROOT)})", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=spark_env(), stdout=f, stderr=subprocess.STDOUT)
        w = Watcher(proc.pid)
        w.start()
        code = proc.wait()
        w.stop_flag.set()
        w.join(timeout=5)
    if code != 0 or not (out / "metrics.json").exists():
        print(f"     FAILED (exit code {code}); see the log")
        return None
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if not w.stages:
        print("     WARNING: could not read the Spark REST API; shuffle numbers are missing")
    finished = datetime.strptime(m["finished_utc"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    boundary = finished - m["seconds_evaluate"] + 1.0      # 1 s margin: finished_utc has whole seconds
    rec = [s for s in w.stages.values() if s.get("completionTime") and utc(s["completionTime"]) <= boundary]
    ev = [s for s in w.stages.values() if s not in rec]
    res = {"method": method, "sample": sample, "cores": 4, "driver_memory": driver_memory,
           "evaluated_users": m["evaluated_users"], "seconds_recommend": m["seconds_recommend"],
           "recommend": sums(rec), "evaluate": sums(ev), "all": sums(list(w.stages.values())),
           "peak_rss_bytes": w.rss_peak, "peak_jvm_rss_bytes": w.jvm_peak, "peak_jvm_heap_bytes": w.heap_peak,
           "machine": {"platform": platform.platform(), "cpu_count": m["env"].get("cpu_count")}}
    (out / "sysmetrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"     shuffle write (recommend) {res['recommend']['shuffle_write_bytes'] / 2**20:.1f} MB, "
          f"peak memory {w.rss_peak / 2**30:.2f} GB")
    return res


def mb(x):
    return f"{x / 2**20:,.1f}"


def gb(x):
    return f"{x / 2**30:.2f}"


def report(methods, samples):
    rows = []
    for m in methods:
        for s in samples:
            f = folder(m, s)[0] / "sysmetrics.json"
            if f.exists():
                rows.append(json.loads(f.read_text(encoding="utf-8")))
    if not rows:
        print("no runs yet")
        return
    lines = ["# Shuffle volume and peak memory (generated by tools/sysmetrics.py)", "",
             f"user_time validation stage, chosen parameters, 4 cores, driver memory {rows[0]['driver_memory']}, "
             f"{rows[0]['machine']['platform']}, {rows[0]['machine']['cpu_count']} CPUs.", "",
             "Shuffle and spill: recommend phase only (evaluation is the same for every method). Peak memory: whole "
             "run, resident memory of the Python driver, the Spark JVM and the Python workers together.", "",
             "| Method | Users | Shuffle write (MB) | Shuffle read (MB) | Spilled to disk (MB) | Peak memory (GB) | "
             "Peak JVM heap (GB) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        rc = r["recommend"]
        lines.append(f"| {r['method']} | {r['sample'] * 100:g}% ({r['evaluated_users']:,}) | {mb(rc['shuffle_write_bytes'])} | "
                     f"{mb(rc['shuffle_read_bytes'])} | {mb(rc['disk_spilled_bytes'])} | {gb(r['peak_rss_bytes'])} | "
                     f"{gb(r['peak_jvm_heap_bytes']) if r['peak_jvm_heap_bytes'] else 'n/a'} |")
    (ROOT / "reports" / "sysmetrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reports" / "sysmetrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print("\n-> reports/sysmetrics.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("methods", nargs="*", help="default: all five")
    ap.add_argument("--samples", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--driver-memory", default="14g", help="use the same value as the scale runs (14g)")
    ap.add_argument("--report", action="store_true", help="only rebuild reports/sysmetrics.md")
    ap.add_argument("--force", action="store_true", help="re-run sizes that already have sysmetrics.json")
    args = ap.parse_args()
    methods = args.methods or METHODS
    if platform.system() == "Windows" and not args.report:
        raise SystemExit("run this on macOS or Linux (it reads process memory with `ps`)")
    if not args.report:
        for m in methods:
            for s in args.samples:
                if (folder(m, s)[0] / "sysmetrics.json").exists() and not args.force:
                    print(f"skip {m} {s * 100:g}% (done)")
                    continue
                run_one(m, s, args.driver_memory)
    report(methods, args.samples)


if __name__ == "__main__":
    main()
