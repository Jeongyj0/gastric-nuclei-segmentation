from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.study import (TISSUES, CONDITIONS, load_config, training_settings, read_manifest,
                       project_path, sha256, source_hashes, json_hash, write_json)
from src.train.layout import require_main_run
from src.train.control import validate_completed_run


def check_run(config, tissue, condition, seed):
    run = project_path(Path("runs/models") / tissue / condition / f"seed_{seed}", must_exist=True)
    resolved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    require_main_run(resolved)
    status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    protocol = json.loads(project_path("runs/protocol.json", must_exist=True).read_text(encoding="utf-8"))
    require_main_run(protocol)
    settings = training_settings(config)
    validate_completed_run(status, settings)
    if settings != resolved["settings"] or resolved["protocol_sha256"] != json_hash(protocol):
        raise ValueError("Shared training settings do not match the training record.")
    if protocol["sources"] != source_hashes():
        raise ValueError("Source code changed after training. Check the evaluation path first.")
    if any(sha256(project_path(path, must_exist=True)) != digest
           for path, digest in protocol["manifests"].items()):
        raise ValueError("Fixed manifests changed after training.")
    if protocol["pretrained_sha256"] != sha256(project_path(config["pretrained"], must_exist=True)):
        raise ValueError("Shared pretrained weights have changed.")
    if (resolved["tissue"], resolved["condition"], resolved["seed"]) != (tissue, condition, seed):
        raise ValueError("Run identity does not match the results directory.")
    return run, resolved, protocol


def evaluate_one(config, tissue, condition, seed, device, output_root):
    import torch
    from src.eval.pipeline import evaluate_images
    from src.train.runtime import create_model
    from src.train.train import write_private_rows
    run, resolved, protocol = check_run(config, tissue, condition, seed)
    rows, manifest = read_manifest(config, tissue, "test")
    if sha256(manifest) != resolved["test_manifest_sha256"]:
        raise ValueError("Test manifest differs from the training-time record.")
    out = project_path(output_root / tissue / condition / f"seed_{seed}")
    if out.exists():
        raise FileExistsError("Evaluation directory already exists; existing results will not be overwritten.")
    checkpoint_path = run / "best_pq.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_main_run(checkpoint)
    if (checkpoint["protocol_sha256"] != resolved["protocol_sha256"] or
            (checkpoint["tissue"], checkpoint["condition"], checkpoint["seed"]) != (tissue, condition, seed) or
            checkpoint["update"] != checkpoint["best"]["pq"]["update"]):
        raise ValueError("Primary evaluation checkpoint has invalid provenance or selection criteria.")
    model = create_model(input_ch=3, nr_types=None, freeze=False, mode="fast")
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    selected_update = int(checkpoint["update"])
    del checkpoint
    out.mkdir(parents=True, exist_ok=False)
    (out / "private").mkdir()
    write_json(out / "status.json", {"status": "running"})
    try:
        results, summary = evaluate_images(model, rows, device, protocol["inference_batch_size"])
        summary.update(tissue=tissue, condition=condition, seed=seed,
                       checkpoint="best_pq.pth", selected_update=selected_update,
                       checkpoint_sha256=sha256(checkpoint_path),
                       protocol_sha256=resolved["protocol_sha256"],
                       initial_state_sha256=resolved["initial_state_sha256"],
                       test_manifest_sha256=sha256(manifest), test_used_in_design=True)
        private_path = out / "private/test_metrics.csv"
        write_private_rows(private_path, results)
        summary["metrics_sha256"] = sha256(private_path)
        write_json(out / "test_summary.json", summary)
        write_json(out / "status.json", {"status": "completed"})
        print(f"{tissue}/{condition}/seed_{seed}: image-mean PQ={summary['mean_pq']:.6f}")
        return summary
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Evaluate selected checkpoints on the fixed test set.")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--all", action="store_true", help="Evaluate all configured tissues, conditions, and seeds")
    parser.add_argument("--cancer", choices=TISSUES)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = training_settings(config)
    if args.all:
        if any(v is not None for v in (args.cancer, args.condition, args.seed)):
            parser.error("Do not combine --all with individual run arguments.")
        targets = [(t, c, s) for t in TISSUES for c in CONDITIONS for s in settings["training_seeds"]]
    else:
        if args.cancer is None or args.condition is None or args.seed is None:
            parser.error("Specify --all or --cancer, --condition, and --seed.")
        if args.seed not in settings["training_seeds"]:
            parser.error("Seed is not listed in the shared configuration.")
        targets = [(args.cancer, args.condition, args.seed)]
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/runtime_cache/matplotlib"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(ROOT / "outputs/runtime_cache/cuda"))
    import torch
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    output_root = project_path(args.output_dir)
    for tissue, condition, seed in targets:
        evaluate_one(config, tissue, condition, seed, device, output_root)


if __name__ == "__main__":
    main()

