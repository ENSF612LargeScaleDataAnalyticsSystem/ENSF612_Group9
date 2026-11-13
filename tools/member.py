"""Group workflow helper: one command per stage of a member's task (tasks/<method>.md).

    python tools/member.py check lsh      # is this machine ready, and is its data identical to the group's?
    python tools/member.py pilot lsh      # re-run the chosen settings on the 10% pilot (validation) and
                                          # compare with the committed numbers: proves the machine reproduces them
    python tools/member.py test lsh       # the final test runs: user_time and global_time, once each
    python tools/member.py push lsh       # commit this member's files and push branch member/lsh

Methods: knn, lsh, ppr, als. The scalability runs (tools/scale.py) are a fifth piece of work with its own
branch: `check scale` and `push scale` work the same way; pilot and test do not apply to it. Works on Windows, macOS and Linux; run it with the project's venv Python
from the project root. Parameters always come from tools/chosen.json (picked on validation), never
from the command line, so nobody can tune on the test part by accident.
"""
import argparse
import ast
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("knn", "lsh", "ppr", "als")
TEST_SPLITS = ("user_time", "global_time")
PILOT_SPLIT = "user_time_10pct"
PILOT_TOLERANCE = 0.002          # allowed |difference| in each metric between two machines

# Files each member may commit. Everything else belongs to the shared foundation or to another member.
OWN = {
    "knn": ["results/knn", "report/knn.md", "tasks/knn.md"],
    "lsh": ["results/lsh", "report/lsh.md", "tasks/lsh.md"],
    "ppr": ["results/ppr", "report/ppr.md", "report/data.md", "tasks/ppr.md"],
    "als": ["results/als", "report/als.md", "report/evaluation.md", "tasks/als.md"],
}
SCALE = "scale"                  # the scalability runs: timing folders of all five methods


def own_paths(method):
    if method != SCALE:
        return OWN[method]
    runs = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "results").glob("*/user_time/val__scale_*"))
    return [*runs, "tasks/scale.md"]
# A change here means the member's numbers were not produced by the group's code.
FROZEN = ("src/", "tests/", "tools/", "requirements.txt")


# ----------------------------------------------------------------------------------------- helpers

def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r


def spark_env():
    """Environment for a Spark run. On Windows, JAVA_HOME / HADOOP_HOME set in the user settings after
    the terminal was opened are not in os.environ yet, so read them from the registry."""
    env = dict(os.environ)
    if platform.system() == "Windows":
        import winreg
        for name in ("JAVA_HOME", "HADOOP_HOME"):
            if not env.get(name):
                try:
                    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                        env[name] = winreg.QueryValueEx(key, name)[0]
                except OSError:
                    pass
    extra = [str(Path(env[n]) / "bin") for n in ("JAVA_HOME", "HADOOP_HOME") if env.get(n)]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def chosen_params(method):
    """tools/chosen.json -> (['--param', 'k=v', ...], {'k': v, ...})."""
    items = json.loads((ROOT / "tools" / "chosen.json").read_text(encoding="utf-8")).get(method, [])
    flags, parsed = [], {}
    for item in items:
        key, value = item.split("=", 1)
        flags += ["--param", item]
        try:
            parsed[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed[key] = value
    return flags, parsed


def run_logged(log_name, py_args, env):
    """Run the venv Python with py_args; output goes to logs/<log_name>.log. Returns the exit code."""
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / f"{log_name}.log"
    print(f"  running: python {' '.join(py_args)}\n  log:     {log}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        code = subprocess.run([sys.executable, *py_args], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\nexit: {code}\n")
    return code


def branch():
    r = git("rev-parse", "--abbrev-ref", "HEAD", check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def committed_json(rel):
    r = git("show", f"HEAD:{rel}", check=False)
    return json.loads(r.stdout) if r.returncode == 0 else None


def run_folder(method, split, stage, tag=""):
    return ROOT / "results" / method / split / (stage + (f"__{tag}" if tag else ""))


# ------------------------------------------------------------------------------------------- check

def checks(method):
    """Yield (ok, label, how to fix)."""
    win = platform.system() == "Windows"
    v = sys.version_info
    if win:
        yield v[:2] == (3, 11), f"Python 3.11 (found {v.major}.{v.minor})", \
            "PySpark 4.0.x crashes on Windows with Python 3.12+. Recreate the venv: py -3.11 -m venv .venv"
    else:
        yield v[:2] >= (3, 10), f"Python 3.10+ (found {v.major}.{v.minor})", "Create the venv with Python 3.10 or newer."
    yield sys.prefix != sys.base_prefix, "running inside the project's virtual environment", \
        r"Use .venv\Scripts\python.exe (Windows) or .venv/bin/python to run this script."
    try:
        import pyspark
        yield pyspark.__version__ == "4.0.1", f"pyspark 4.0.1 (found {pyspark.__version__})", "pip install -r requirements.txt"
    except ImportError:
        yield False, "pyspark installed", "pip install -r requirements.txt"

    env = spark_env()
    java = shutil.which("java", path=env["PATH"])
    yield bool(java), f"Java found ({java})", "Install JDK 17 (Eclipse Temurin) and set JAVA_HOME. See README section 1."
    if win:
        hh = Path(env.get("HADOOP_HOME", "")) / "bin"
        ok = bool(env.get("HADOOP_HOME")) and (hh / "winutils.exe").exists() and (hh / "hadoop.dll").exists()
        yield ok, r"HADOOP_HOME\bin has winutils.exe and hadoop.dll", \
            r"Put winutils.exe and hadoop.dll (Hadoop 3.x) in C:\hadoop\bin and set HADOOP_HOME=C:\hadoop. See README section 1."

    b = branch()
    yield b == f"member/{method}", f"git branch is member/{method} (found {b})", \
        f"git checkout -b member/{method}   (or: git checkout member/{method} if it already exists)"
    yield git("remote", "get-url", "origin", check=False).returncode == 0, "git remote 'origin' is set", \
        "Clone the group repository instead of copying the folder."
    dirty = [p for _, p in changed_files() if p.startswith(FROZEN)]
    yield not dirty, "shared code is unchanged (src/, tests/, tools/, requirements.txt)", \
        "Undo the changes: git checkout -- " + " ".join(dirty)

    raw = ROOT / "data" / "raw" / "ml-20m"
    yield (raw / "ratings.csv").exists() and (raw / "movies.csv").exists(), "data/raw/ml-20m has ratings.csv and movies.csv", \
        "Download ml-20m.zip from https://grouplens.org/datasets/movielens/20m/ and unzip it into data/raw/. See README section 2."
    missing = [f"{s}/{p}" for s in (*TEST_SPLITS, PILOT_SPLIT) for p in ("train", "val", "test")
               if not (ROOT / "data" / "splits" / s / p / "_SUCCESS").exists()]
    yield not missing, "data/splits is complete", \
        "Run: python -m src.ingest   then: python -m src.split   (missing: " + ", ".join(missing[:3]) + ")"

    # ingest and split rewrite these two files, so after a member's own run they describe that member's
    # data. They must equal the copies committed by the group: same rows in train / val / test everywhere.
    for rel, step in (("reports/data_profile.json", "src.ingest"), ("reports/split_stats.json", "src.split")):
        ref, local = committed_json(rel), None
        if (ROOT / rel).exists():
            local = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        yield ref is not None and local == ref, f"{rel} is identical to the group's copy", \
            f"Your data differs from the group's. Do not run tests. Re-download ml-20m and re-run python -m {step}; " \
            "if it still differs, tell the group."


def cmd_check(method, _args):
    failed = 0
    for ok, label, fix in checks(method):
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failed += 1
            print(f"       fix: {fix}")
    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------- pilot

def reference_run(method, params):
    """The committed validation run on the pilot whose parameters equal the chosen ones."""
    for folder in sorted((ROOT / "results" / method / PILOT_SPLIT).glob("val*")):
        m = folder / "metrics.json"
        if folder.name != "val__repro" and m.exists():
            j = json.loads(m.read_text(encoding="utf-8"))
            if j.get("params") == params and j.get("sample", 1.0) == 1.0:
                return j
    return None


def cmd_pilot(method, args):
    flags, params = chosen_params(method)
    ref = reference_run(method, params)
    if ref is None:
        sys.exit(f"No committed pilot run of {method} with the chosen parameters {params}. Tell the group.")
    extra = ["--driver-memory", args.driver_memory] if args.driver_memory else []
    code = run_logged(f"pilot_repro_{method}", ["-m", "src.run", "--method", method, "--split", PILOT_SPLIT,
                                                "--stage", "val", "--tag", "repro", "--n-boot", "0", *flags, *extra], spark_env())
    out = run_folder(method, PILOT_SPLIT, "val", "repro")
    if code != 0 or not (out / "metrics.json").exists():
        shutil.rmtree(out, ignore_errors=True)
        print(f"[FAIL] pilot run exited with {code}; read logs/pilot_repro_{method}.log\n"
              "       This is a crash, not a wrong number: close other heavy programs and run the command again.")
        return 1
    mine = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    shutil.rmtree(out)                               # a check, not a result: keep it out of the tables
    bad = 0
    print(f"\n{method} on {PILOT_SPLIT} (validation), parameters {params}")
    for key in ("precision@10", "recall@10", "ndcg@10", "coverage"):
        diff = abs(mine[key] - ref[key])
        ok = diff <= PILOT_TOLERANCE and mine["evaluated_users"] == ref["evaluated_users"]
        bad += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {key:<13} here {mine[key]:.6f}   group {ref[key]:.6f}   diff {diff:.6f}")
    print(f"       recommend time here {mine['seconds_recommend']:.0f}s, on the group's machine {ref['seconds_recommend']:.0f}s")
    if bad:
        print("\nThis machine does not reproduce the group's validation numbers. Do NOT run the test stage. Tell the group.")
    else:
        print("\nPILOT REPRODUCED: this machine gives the group's numbers.")
    return 1 if bad else 0


# -------------------------------------------------------------------------------------------- test

def cmd_test(method, args):
    b = branch()
    if b != f"member/{method}":
        sys.exit(f"You are on branch {b}; switch to member/{method} first.")
    flags, params = chosen_params(method)
    extra = ["--driver-memory", args.driver_memory] if args.driver_memory else []
    env = spark_env()
    print(f"{method}: final test runs with the parameters chosen on validation: {params}")
    for split in TEST_SPLITS:
        out = run_folder(method, split, "test")
        if (out / "metrics.json").exists():
            j = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
            print(f"[DONE] {split}: already run on {j['finished_utc']} UTC (NDCG@10 {j['ndcg@10']:.4f}). "
                  "The test stage runs once; not running it again.")
            continue
        code = run_logged(f"final_{method}_{split}", ["-m", "src.run", "--method", method, "--split", split,
                                                      "--stage", "test", *flags, *extra], env)
        if code != 0 or not (out / "metrics.json").exists():
            print(f"[FAIL] {split}: exit {code}, no result was produced. Read the end of logs/final_{method}_{split}.log.\n"
                  "       Nothing was scored, so running this command again is allowed (it skips finished splits).\n"
                  "       On a machine with 8 GB of RAM add: --driver-memory 5g")
            return 1
        j = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        print(f"[DONE] {split}: P@10 {j['precision@10']:.4f}  R@10 {j['recall@10']:.4f}  NDCG@10 {j['ndcg@10']:.4f}  "
              f"coverage {j['coverage']:.4f}  ({j['evaluated_users']:,} users, recommend {j['seconds_recommend']:.0f}s)")
    print("\nTEST RUNS COMPLETE. Next: python tools/member.py push " + method)
    return 0


# -------------------------------------------------------------------------------------------- push

def changed_files():
    """[(status, path)] for every modified or untracked file."""
    out = git("status", "--porcelain", "--untracked-files=all", check=False).stdout
    files = []
    for line in out.splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        files.append((line[:2], path))
    return files


def cmd_push(method, args):
    b = branch()
    if b != f"member/{method}":
        sys.exit(f"You are on branch {b}; switch to member/{method} first.")
    own = own_paths(method)
    mine = lambda p: any(p == o or p.startswith(o + "/") for o in own)
    others = [p for _, p in changed_files() if not mine(p)]
    frozen = [p for p in others if p.startswith(FROZEN)]
    if frozen:
        print("[STOP] Shared code was changed on this machine, so these results are not comparable with the group's:")
        for p in frozen:
            print(f"         {p}")
        print("       Undo with: git checkout -- <file>. If the change is a real bug fix, tell the group; the fix\n"
              "       goes to main first and every member pulls it before testing.")
        return 1

    print("Deliverables:")
    if method == SCALE:
        done = sum((ROOT / o / "metrics.json").exists() for o in own if o.startswith("results/"))
        print(f"  [{'x' if done == 30 else ' '}] scalability runs with a result: {done} of 30")
    for split in (TEST_SPLITS if method != SCALE else ()):
        ok = (run_folder(method, split, "test") / "metrics.json").exists()
        print(f"  [{'x' if ok else ' '}] results/{method}/{split}/test/metrics.json")
    for rel in [o for o in own if o.endswith(".md") and o.startswith("report/")]:
        print(f"  [{'x' if (ROOT / rel).exists() else ' '}] {rel}")

    existing = [o for o in own if (ROOT / o).exists()]
    git("add", "--", *existing)
    if git("diff", "--cached", "--quiet", check=False).returncode != 0:
        default = "Scalability runs (RQ3)" if method == SCALE else f"{method}: test results and report section"
        git("commit", "-m", args.message or default)
        print("committed: " + git("log", "-1", "--oneline").stdout.strip())
    else:
        print("nothing new to commit")
    if others:
        print("not included (not this member's files): " + ", ".join(others[:8]) + (" ..." if len(others) > 8 else ""))

    r = git("push", "-u", "origin", f"member/{method}", check=False)
    print((r.stdout + r.stderr).strip())
    if r.returncode != 0:
        print("\n[FAIL] push failed. If it asks for a login, sign in to GitHub with an account that can write to the\n"
              "       group repository, then run this command again. Nothing is lost: the commit is saved locally.")
        return 1
    url = git("remote", "get-url", "origin").stdout.strip()
    if url.endswith(".git"):
        url = url[:-4]
    print(f"\nPUSHED. Open the pull request here:\n  {url}/compare/main...member/{method}?expand=1")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "pilot", "test", "push"])
    ap.add_argument("method", choices=[*METHODS, SCALE])
    ap.add_argument("--driver-memory", help="Spark heap for pilot/test, e.g. 5g on an 8 GB laptop (default 8g)")
    ap.add_argument("-m", "--message", help="commit message for push")
    args = ap.parse_args()
    if args.method == SCALE and args.command in ("pilot", "test"):
        ap.error("scale has no pilot or test step; run: python tools/scale.py")
    sys.exit({"check": cmd_check, "pilot": cmd_pilot, "test": cmd_test, "push": cmd_push}[args.command](args.method, args))


if __name__ == "__main__":
    main()
