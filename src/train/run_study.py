import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from src.study import ROOT, TISSUES, CONDITIONS, load_config, training_settings, project_path, write_json


def main():
    parser = argparse.ArgumentParser(description="Plan or run the configured study.")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--run", action="store_true", help="Start training instead of printing the plan")
    parser.add_argument("--skip-completed", action="store_true", help="Skip completed runs with the same configuration")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = training_settings(config)
    execution = config.get("main_execution", {})
    if args.num_workers is None:
        args.num_workers = execution.get("num_workers", 4)
    if args.inference_batch_size is None:
        args.inference_batch_size = execution.get("inference_batch_size", 1)
    if args.num_workers < 0 or args.inference_batch_size < 1:
        parser.error("Invalid worker count or inference batch size.")
    targets = [(t, c, s) for s in settings["training_seeds"] for t in TISSUES for c in CONDITIONS]
    commands = []
    selected_targets = []
    for tissue, condition, seed in targets:
        directory = project_path(Path("runs/models") / tissue / condition / f"seed_{seed}")
        if directory.exists():
            if args.skip_completed:
                from src.eval.evaluate import check_run
                check_run(config, tissue, condition, seed)
                print(f"Verified completed run; skipping: {tissue}/{condition}/seed_{seed}")
                continue
            raise FileExistsError("Run directory already exists. Use --skip-completed to verify completed runs.")
        command = [sys.executable, "-X", "utf8", "-B", "-m", "src.train.train", "--config", args.config,
                   "--cancer", tissue, "--condition", condition, "--seed", str(seed),
                   "--num-workers", str(args.num_workers), "--inference-batch-size", str(args.inference_batch_size)]
        commands.append(command)
        selected_targets.append({"tissue": tissue, "condition": condition, "seed": seed})
    print(f"Runs: {len(commands)}. Mode: {'training' if args.run else 'plan'}", flush=True)
    if not args.run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    lock = project_path("runs/study.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    started = time.perf_counter()
    status_path = project_path("runs/study_status.json")
    completed = []
    active = None
    try:
        for active, command in zip(selected_targets, commands):
            write_json(status_path, {"status": "running", "pid": os.getpid(),
                       "total_runs": len(commands), "completed_runs": len(completed),
                       "active": active, "completed": completed,
                       "elapsed_seconds": time.perf_counter() - started})
            subprocess.run(command, cwd=ROOT, check=True)
            completed.append(active)
        write_json(status_path, {"status": "completed", "pid": os.getpid(),
                   "total_runs": len(commands), "completed_runs": len(completed),
                   "active": None, "completed": completed,
                   "elapsed_seconds": time.perf_counter() - started})
    except BaseException as error:
        write_json(status_path, {"status": "failed", "pid": os.getpid(),
                   "total_runs": len(commands), "completed_runs": len(completed),
                   "active": active, "completed": completed, "error_type": type(error).__name__,
                   "elapsed_seconds": time.perf_counter() - started})
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
