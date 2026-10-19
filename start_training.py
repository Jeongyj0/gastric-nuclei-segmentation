import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from src.study import load_config, training_settings


ROOT = Path(__file__).resolve().parent


def run_logged(command, log_path):
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed (exit {code}). Log: {log_path.relative_to(ROOT)}")


def run_checks_and_training(attempt):
    base = [sys.executable, "-X", "utf8", "-B", "-u", "-m"]
    relative = attempt.relative_to(ROOT).as_posix()
    print("1/3: Checking batch size 2 and 4 workers.", flush=True)
    run_logged(base + ["src.checks.batch_probe", "--batch-size", "2", "--num-workers", "4",
                      "--output-dir", relative + "/batch_check"], attempt / "batch_check.log")
    batch = json.loads((attempt / "batch_check/result.json").read_text(encoding="utf-8"))
    if (not batch.get("passed") or batch.get("batch_size") != 2 or
            batch.get("free_device_mib_after_probe", 0) < 512):
        raise RuntimeError("Batch/worker or memory check failed. Training not started.")
    print("2/3: Checking code, manifests, regression tests, and CUDA.", flush=True)
    run_logged(base + ["src.checks.preflight", "--runtime", "--output-dir", relative + "/preflight"],
               attempt / "preflight.log")
    preflight = json.loads((attempt / "preflight/preflight.json").read_text(encoding="utf-8"))
    if not preflight.get("ready_for_main_training"):
        raise RuntimeError("Preflight failed. Training not started.")
    print("3/3: Starting 20 training runs: batch size 2, 4 workers, seed 42.", flush=True)
    print(f"Training log: {relative}/training.log", flush=True)
    run_logged(base + ["src.train.run_study", "--run", "--num-workers", "4",
                      "--inference-batch-size", "1"], attempt / "training.log")
    print("All training runs completed.", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the environment and launch the training study.")
    parser.add_argument("--plan-only", action="store_true", help="Print the plan without GPU checks or training")
    args = parser.parse_args(argv)
    config = load_config()
    training_settings(config)
    if args.plan_only:
        subprocess.run([sys.executable, "-X", "utf8", "-B", "-m", "src.train.run_study"], cwd=ROOT, check=True)
        return 0
    if config["batch_size"] != 2 or config["main_execution"]["num_workers"] != 4:
        raise ValueError("Expected batch size 2 and 4 workers.")
    for name in ("models", "protocol.json", "study.lock", "initialization"):
        if (ROOT / "runs" / name).exists():
            raise FileExistsError(f"Existing runs/{name} must be reviewed; refusing to overwrite.")
    lock = ROOT / "runs/launch.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError:
        raise RuntimeError("Launch lock exists; refusing duplicate execution.") from None
    try:
        output = ROOT / "outputs/training"
        output.mkdir(parents=True, exist_ok=True)
        for number in itertools.count(1):
            attempt = output / f"attempt_{number}"
            try:
                attempt.mkdir()
                break
            except FileExistsError:
                continue
        shutil.copyfile(ROOT / "configs/study.json", attempt / "study_config.json")
        shutil.copyfile(Path(__file__), attempt / "launcher_source.py.saved")
        run_checks_and_training(attempt)
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Launch or execution stopped: {error}", file=sys.stderr)
        sys.exit(1)
