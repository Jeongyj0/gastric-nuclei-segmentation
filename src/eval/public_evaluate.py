"""Evaluate released state dictionaries without private training-run records."""
import argparse
import gc
from pathlib import Path

from src.study import TISSUES, CONDITIONS, ROOT, project_path, read_manifest, sha256, source_hashes, write_json
from src.eval.predict import load_public_model


CHECKPOINT_KIND = "public_state_dict"


def evaluation_sources():
    return {**source_hashes(), "analysis/froc.py": sha256(ROOT / "analysis/froc.py")}


def evaluate_one(rows, manifest, tissue, condition, seed, device, checkpoint_root,
                 output_root, *, batch_size=1, froc=False):
    """Rows are validated by read_manifest in the CLI; tests may supply small fixtures."""
    from src.eval.pipeline import evaluate_images
    from src.train.train import write_private_rows
    import torch

    out = project_path(Path(output_root) / tissue / condition / f"seed_{seed}")
    if out.exists() or out.is_symlink():
        raise FileExistsError("Evaluation directory exists; select a new output directory.")
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    manifest_hash = sha256(manifest)
    checkpoint_manifest_hash = sha256(checkpoint_root / "manifest.json")
    model, metadata = load_public_model(tissue, condition, seed, device=device,
                                        checkpoint_root=checkpoint_root)
    input_records, froc_directories = [], []

    def observe(index, row, truth, predicted, raw, scores):
        input_records.append({key: row[key] for key in ("id", "pid", "image_path", "inst_mask_path")})
        input_records[-1].update(
            image_sha256=sha256(project_path(row["image_path"], must_exist=True)),
            mask_sha256=sha256(project_path(row["inst_mask_path"], must_exist=True)))
        if froc:
            from analysis.froc import score_and_match, audit_operating_point, save_private_result
            result = score_and_match(truth, predicted, raw[..., 0])
            audit = audit_operating_point(result, scores)
            folder = out / "private" / f"image_{index:04d}"
            save_private_result(folder, result, audit, provenance={
                "scope": "current_public_evaluation", "checkpoint_sha256": metadata["sha256"],
                "test_manifest_sha256": manifest_hash, **input_records[-1]})
            froc_directories.append(folder)

    created_output = False
    try:
        out.mkdir(parents=True, exist_ok=False)
        created_output = True
        (out / "private").mkdir()
        write_json(out / "status.json", {"status": "running"})
        results, summary = evaluate_images(model, rows, device, batch_size, on_prediction=observe)
        if sha256(manifest) != manifest_hash or sha256(checkpoint_root / "manifest.json") != checkpoint_manifest_hash:
            raise ValueError("An input manifest changed during evaluation.")
        metrics_path = out / "private/test_metrics.csv"
        inputs_path = out / "private/inputs.json"
        write_private_rows(metrics_path, results)
        write_json(inputs_path, input_records)
        summary.update(tissue=tissue, condition=condition, seed=seed,
                       checkpoint=CHECKPOINT_KIND, public_model=metadata,
                       checkpoint_sha256=metadata["sha256"], selected_update=metadata["selected_update"],
                       checkpoint_manifest_sha256=checkpoint_manifest_hash,
                       test_manifest_sha256=manifest_hash, metrics_sha256=sha256(metrics_path),
                       input_records_sha256=sha256(inputs_path), evaluation_sources=evaluation_sources(),
                       device=str(device), inference_batch_size=batch_size,
                       scope="Public-weight evaluation on the supplied test manifest; original run provenance is not verified")
        if froc:
            from analysis.froc import aggregate_condition
            summary["froc"] = aggregate_condition(froc_directories, tissue, condition, out, expected_n=len(rows))
            # The no-prediction curve has only its +infinity threshold. Preserve
            # that meaning using a string, rather than nonstandard JSON Infinity.
            if summary["froc"]["score"] == float("inf"):
                summary["froc"]["score"] = "inf"
            summary["froc_audit_scope"] = "Agreement with this evaluation's metrics, not a historical experiment"
        write_json(out / "test_summary.json", summary)
        write_json(out / "status.json", {"status": "completed"})
        print(f"{tissue}/{condition}/seed_{seed}: image-mean PQ={summary['mean_pq']:.6f}")
        return summary
    except BaseException as error:
        if created_output:
            write_json(out / "status.json", {"status": "failed", "error_type": type(error).__name__})
        raise
    finally:
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--tissue", choices=TISSUES)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--froc", action="store_true")
    parser.add_argument("--output-dir", default="outputs/public_evaluation")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.batch_size < 1:
        parser.error("Seed must be nonnegative and batch size positive.")
    if args.all:
        if args.tissue is not None or args.condition is not None:
            parser.error("Do not combine --all with --tissue or --condition.")
        targets = [(t, c) for t in TISSUES for c in CONDITIONS]
    else:
        if args.tissue is None or args.condition is None:
            parser.error("Specify --all or both --tissue and --condition.")
        targets = [(args.tissue, args.condition)]
    config = {"manifest_root": args.manifest_root}
    for tissue, condition in targets:
        rows, manifest = read_manifest(config, tissue, "test")
        evaluate_one(rows, manifest, tissue, condition, args.seed, args.device,
                     args.checkpoint_dir, project_path(args.output_dir),
                     batch_size=args.batch_size, froc=args.froc)


if __name__ == "__main__":
    main()
