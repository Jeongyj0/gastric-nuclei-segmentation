from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.study import (TISSUES, CONDITIONS, execution_policy, load_config, training_settings,
                       read_manifest, audit_manifests, project_path, source_hashes,
                       sha256, json_hash, write_json)
from src.train.control import validation_due, select_checkpoints, ValidationPlateau
from src.train.layout import run_layout


def lock_record(path, record):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError("Configuration or initial state differs from the existing study record. Preserve existing results.")


def write_private_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args, config, settings, audit, *, pilot_id=None):
    run_started = time.perf_counter()
    layout = run_layout(args.cancer, args.condition, args.seed, pilot_id)
    scope, out = layout["scope"], layout["out"]
    identity = {"phase": layout["phase"], "pilot_id": pilot_id}
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/runtime_cache/matplotlib"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "outputs/runtime_cache/cuda"))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from src.data.hovernet_dataset import HoverNetTrainDataset
    from src.eval.pipeline import evaluate_images
    from src.train.runtime import (set_seed, seed_worker, create_model, load_pretrained,
                                   model_hash, training_mode, compute_loss, save_checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a working CUDA environment.")
    device = torch.device("cuda")
    if out.exists():
        raise FileExistsError("Run directory for this tissue/condition/seed already exists; refusing to overwrite.")
    train_rows, train_path = read_manifest(config, args.cancer, "train", args.condition)
    val_rows, val_path = read_manifest(config, args.cancer, "val")
    dataset = HoverNetTrainDataset(pd.DataFrame(train_rows), seed=args.seed)
    if settings["batch_size"] > len(dataset):
        raise ValueError("Batch size exceeds the number of training patches.")
    policy = execution_policy(settings)
    protocol = {"settings": settings, "policy": policy, "sources": source_hashes(),
                "manifests": audit["manifest_sha256"],
                "pretrained_sha256": sha256(project_path(config["pretrained"])),
                "inference_batch_size": args.inference_batch_size, "num_workers": args.num_workers}
    if pilot_id is not None:
        protocol.update(identity)
    protocol_hash = json_hash(protocol)
    set_seed(args.seed)
    model = create_model(input_ch=3, nr_types=None, freeze=False, mode="fast")
    pretrained_info = load_pretrained(model, project_path(config["pretrained"]))
    initial_hash = model_hash(model)
    lock_record(scope / "protocol.json", protocol)
    lock_record(scope / f"initialization/seed_{args.seed}.json",
                {"seed": args.seed, "initial_state_sha256": initial_hash,
                 "pretrained_sha256": protocol["pretrained_sha256"]})
    model = model.to(device)
    lr = settings["lr_schedule"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr["initial_lr"], betas=(0.9, 0.999))
    adaptive = lr["kind"] == "plateau_once"
    scheduler = (ValidationPlateau(optimizer, lr, settings["early_stopping"]) if adaptive else
                 torch.optim.lr_scheduler.StepLR(
                     optimizer, step_size=lr["step_size_updates"], gamma=lr["gamma"]))
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=settings["batch_size"], shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          generator=generator, worker_init_fn=seed_worker, persistent_workers=False)
    if args.num_workers:
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_options)
    out.mkdir(parents=True, exist_ok=False)
    private = out / "private"
    private.mkdir()
    environment = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                   "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                   "cudnn": torch.backends.cudnn.version(),
                   "deterministic_cudnn": True, "strict_deterministic_algorithms": False}
    resolved = {**identity, "tissue": args.cancer, "condition": args.condition, "seed": args.seed,
                "settings": settings, "policy": policy, "protocol_sha256": protocol_hash,
                "initial_state_sha256": initial_hash, "pretrained": pretrained_info,
                "train_manifest_sha256": sha256(train_path), "val_manifest_sha256": sha256(val_path),
                "test_manifest_sha256": audit["manifest_sha256"][
                    f"{config['manifest_root']}/{args.cancer}/test.csv"],
                "train_images": len(train_rows), "validation_images": len(val_rows),
                "patches_per_pass": len(dataset), "batches_per_pass": len(loader),
                "dropped_patches_per_complete_pass": len(dataset) % settings["batch_size"],
                "environment": environment}
    write_json(out / "config.json", resolved)
    write_json(out / "status.json", {**identity, "status": "running", "update": 0})
    log_fields = ["update", "epoch", "steps_in_epoch", "train_total", "np_ce", "np_dice",
                  "hv_mse", "hv_msge", "lr_used", "lr_next", "real_patches", "synthetic_patches",
                  "train_seconds", "validation_seconds", "checkpoint_seconds", "elapsed_seconds", "peak_gpu_mib",
                  "val_mean_pq", "val_mean_dq", "val_mean_sq", "val_mean_np_dice",
                  "val_pooled_np_dice", "val_instance_precision", "val_instance_recall",
                  "val_pred_instances", "val_gt_instances", "val_empty_pred",
                  "base_equivalent_epochs", "bad_checks", "lr_reduced", "stop_requested"]
    with (out / "log.csv").open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=log_fields).writeheader()
    update = epoch = 0
    maximum = settings["max_optimizer_updates"]
    interval = settings["validation_interval_updates"]
    freeze_updates = settings["encoder_freeze_schedule"]["freeze_updates"]
    best = {"pq": None, "np_dice": None}
    exposures = {"real": 0, "synthetic": 0}
    sums = {k: 0.0 for k in ("train_total", "np_ce", "np_dice", "hv_mse", "hv_msge")}
    interval_steps = 0
    train_seconds = 0.0
    started = time.perf_counter()
    initialization_seconds = started - run_started
    torch.cuda.reset_peak_memory_stats()
    label = "Pilot" if pilot_id is not None else "Main"
    should_stop = False
    def current_status(stage):
        return {**identity, "status": "running", "stage": stage, "update": update,
                "maximum_updates": maximum, "batch_size": settings["batch_size"],
                "base_equivalent_epochs": update * settings["batch_size"] / 2400,
                "learning_rate": optimizer.param_groups[0]["lr"], "best": best,
                "exposures": dict(exposures), "seconds": time.perf_counter() - started,
                "validation_control": scheduler.state_dict() if adaptive else None}
    print(f"{label} training started: {args.cancer}/{args.condition}/seed_{args.seed}, "
          f"batch {settings['batch_size']}, up to {maximum} updates, validation every {interval}", flush=True)
    try:
        while update < maximum and not should_stop:
            dataset.set_epoch(epoch)
            training_mode(model, update < freeze_updates)
            segment_started = time.perf_counter()
            for steps_in_epoch, batch in enumerate(loader, 1):
                if update == freeze_updates:
                    training_mode(model, False)
                images = batch["img"].to(device, non_blocking=True)
                true_np = batch["np_map"].to(device, non_blocking=True)
                true_hv = batch["hv_map"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss, terms = compute_loss(model(images), true_np, true_hv)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("Training loss is not finite.")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                lr_used = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if not adaptive:
                    scheduler.step()
                update += 1
                interval_steps += 1
                sums["train_total"] += float(loss.detach())
                for key, value in terms.items():
                    sums[key] += float(value.detach())
                for source in batch["source"]:
                    exposures[source] += 1
                del loss, terms, images, true_np, true_hv, batch, norm
                if update == 1 or update % 50 == 0:
                    write_json(out / "status.json", current_status("training"))
                    print(f"  update {update}/{maximum}", flush=True)
                if validation_due(update, maximum, interval):
                    torch.cuda.synchronize()
                    train_seconds += time.perf_counter() - segment_started
                    write_json(out / "status.json", current_status("validation"))
                    model.eval()
                    val_rows_out, validation = evaluate_images(
                        model, val_rows, device, args.inference_batch_size)
                    selected = select_checkpoints(
                        best, validation["mean_pq"], validation["pooled_np_dice"], update)
                    previous_lr = optimizer.param_groups[0]["lr"]
                    if adaptive:
                        should_stop = scheduler.step(validation["mean_pq"], update)
                    if optimizer.param_groups[0]["lr"] != previous_lr:
                        print(f"  LR reduced: {previous_lr:g} -> {optimizer.param_groups[0]['lr']:g}", flush=True)
                    write_private_rows(private / f"validation_{update:09d}.csv", val_rows_out)
                    log = {"update": update, "epoch": epoch + 1, "steps_in_epoch": steps_in_epoch,
                           **{key: value / interval_steps for key, value in sums.items()},
                           "lr_used": lr_used, "lr_next": optimizer.param_groups[0]["lr"],
                           "real_patches": exposures["real"], "synthetic_patches": exposures["synthetic"],
                           "train_seconds": train_seconds, "validation_seconds": validation["seconds"],
                           "elapsed_seconds": time.perf_counter() - started,
                           "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                           "base_equivalent_epochs": update * settings["batch_size"] / 2400,
                           "bad_checks": scheduler.bad_checks if adaptive else 0,
                           "lr_reduced": scheduler.lr_reduced if adaptive else False,
                           "stop_requested": should_stop}
                    for key in ("mean_pq", "mean_dq", "mean_sq", "mean_np_dice", "pooled_np_dice",
                                "instance_precision", "instance_recall", "pred_instances",
                                "gt_instances", "empty_pred"):
                        log["val_" + key] = validation[key]
                    metadata = {**identity, "update": update, "epoch": epoch + 1,
                                "steps_in_epoch": steps_in_epoch, "best": best,
                                "validation": validation, "exposures": dict(exposures),
                                "tissue": args.cancer, "condition": args.condition, "seed": args.seed,
                                "protocol_sha256": protocol_hash, "initial_state_sha256": initial_hash,
                                "loader_generator_state": generator.get_state(),
                                "validation_control": scheduler.state_dict() if adaptive else None,
                                "stop_requested": should_stop}
                    checkpoint_started = time.perf_counter()
                    for filename in [*selected, "last.pth"]:
                        save_checkpoint(out / filename, model, optimizer, scheduler, metadata)
                    log["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
                    log["elapsed_seconds"] = time.perf_counter() - started
                    with (out / "log.csv").open("a", newline="", encoding="utf-8") as stream:
                        csv.DictWriter(stream, fieldnames=log_fields).writerow(log)
                    write_json(out / "status.json", current_status("training"))
                    print(f"  validation PQ={validation['mean_pq']:.5f}, "
                          f"NP Dice={validation['pooled_np_dice']:.5f}, "
                          f"checks without improvement {scheduler.bad_checks if adaptive else 0}", flush=True)
                    sums = dict.fromkeys(sums, 0.0)
                    interval_steps = 0
                    training_mode(model, update < freeze_updates)
                    segment_started = time.perf_counter()
                if update >= maximum or should_stop:
                    break
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - segment_started
            epoch += 1
        stop_reason = "early_stopping" if should_stop else "max_optimizer_updates"
        write_json(out / "status.json", {**current_status("finished"), "status": "completed",
                   "stop_reason": stop_reason,
                   "exposures": exposures, "seconds": time.perf_counter() - started,
                   "initialization_seconds": initialization_seconds,
                   "total_seconds": time.perf_counter() - run_started})
        print(f"Training completed: {update} updates ({stop_reason}). Main checkpoint: best_pq.pth", flush=True)
    except BaseException as error:
        (private / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {**identity, "status": "failed", "update": update,
                   "error_type": type(error).__name__, "best": best, "exposures": exposures,
                   "seconds": time.perf_counter() - started, "initialization_seconds": initialization_seconds,
                   "total_seconds": time.perf_counter() - run_started})
        raise RuntimeError("Training stopped. See private/failure_traceback.txt in the run directory.") from None


def main():
    parser = argparse.ArgumentParser(description="Train one tissue, condition, and seed.")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--cancer", required=True, choices=TISSUES)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--inference-batch-size", type=int)
    parser.add_argument("--check-config", action="store_true", help="Check configuration without training")
    args = parser.parse_args()
    config = load_config(args.config)
    execution = config.get("main_execution", {})
    if args.num_workers is None:
        args.num_workers = execution.get("num_workers", 4)
    if args.inference_batch_size is None:
        args.inference_batch_size = execution.get("inference_batch_size", 1)
    try:
        settings = training_settings(config)
        if args.seed not in settings["training_seeds"]:
            raise ValueError("Seed is not listed in training_seeds.")
        if args.num_workers < 0 or args.inference_batch_size < 1:
            raise ValueError("Worker count must be nonnegative and inference batch size must be positive.")
    except ValueError as error:
        parser.error(str(error))
    audit = audit_manifests(config)
    if args.check_config:
        print("Configuration, manifest, and patient separation checks passed. Training not started.")
        return
    run(args, config, settings, audit)


if __name__ == "__main__":
    main()

