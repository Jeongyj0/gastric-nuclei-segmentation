import argparse
import os
import sys
import time
import traceback
from src.study import ROOT, TISSUES, load_config, project_path, read_manifest, write_json


def probe(batch_size, num_workers):
    import pandas as pd
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Subset
    from src.data.hovernet_dataset import HoverNetTrainDataset
    from src.train.runtime import (set_seed, seed_worker, create_model, load_pretrained,
                                   training_mode, compute_loss)
    config = load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    set_seed(config["training_seeds"][0])
    datasets = []
    for tissue in TISSUES:
        for condition in ("base", "replace100"):
            rows, _ = read_manifest(config, tissue, "train", condition)
            dataset = HoverNetTrainDataset(pd.DataFrame(rows), seed=config["training_seeds"][0])
            datasets.append(Subset(dataset, [index * 4 for index in range(batch_size)]))
    loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(config["training_seeds"][0]))
    model = create_model(input_ch=3, nr_types=None, freeze=False, mode="fast")
    load_pretrained(model, project_path(config["pretrained"]))
    model = model.cuda()
    training_mode(model, False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    torch.cuda.reset_peak_memory_stats()
    steps, losses, durations = 0, [], []
    for batch in loader:
        torch.cuda.synchronize()
        started = time.perf_counter()
        images = batch["img"].cuda(non_blocking=True)
        np_map = batch["np_map"].cuda(non_blocking=True)
        hv_map = batch["hv_map"].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = compute_loss(model(images), np_map, hv_map)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Probe loss is not finite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
        steps += 1
        print(f"Batch {batch_size}: training step {steps}/8", flush=True)
        del loss, terms, images, np_map, hv_map, batch
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 256, 256), device="cuda"))
    if any(not torch.isfinite(value).all().item() for value in output.values()):
        raise FloatingPointError("Inference output is not finite.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {"passed": True, "batch_size": batch_size, "optimizer_steps": steps,
            "tissue_source_groups": 8, "all_losses_and_gradients_finite": True,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "free_device_mib_after_probe": free_bytes / 2**20,
            "total_device_mib": total_bytes / 2**20,
            "step_seconds_excluding_data_loading": durations,
            "checkpoint_saved": False, "main_training_executed": False,
            "disposable_model": True}


def main():
    parser = argparse.ArgumentParser(description="Probe CUDA training and inference for a batch size.")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.batch_size not in (1, 2, 4) or args.num_workers < 0:
        parser.error("Batch size must be 1, 2, or 4; worker count must be nonnegative.")
    out = project_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "private").mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/runtime_cache/matplotlib"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "outputs/runtime_cache/cuda"))
    try:
        result = probe(args.batch_size, args.num_workers)
        code = 0
    except Exception as error:
        import torch
        result = {"passed": False, "batch_size": args.batch_size,
                  "error_type": type(error).__name__,
                  "out_of_memory": isinstance(error, torch.cuda.OutOfMemoryError),
                  "main_training_executed": False, "checkpoint_saved": False}
        (out / "private/failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        code = 2
    write_json(out / "result.json", result)
    print("Batch probe:", "passed" if result["passed"] else "failed", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
