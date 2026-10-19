"""Paired patient bootstrap for evaluations of the released model manifest."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.study import ROOT, TISSUES, CONDITIONS, project_path, read_manifest, sha256, write_json
from src.eval.predict import _manifest_entry
from src.eval.public_evaluate import CHECKPOINT_KIND
from src.eval.statistics import CONTRASTS, load_evaluation, paired_patient_bootstrap


def summarize_public(evaluation_root, manifest_root, checkpoint_root, config_path, output_root):
    config = json.loads(project_path(config_path, must_exist=True).read_text(encoding="utf-8-sig"))
    bootstrap = config.get("bootstrap", {})
    repetitions, seed = bootstrap.get("repetitions"), bootstrap.get("seed")
    if (type(repetitions) is not int or repetitions < 1 or type(seed) is not int or seed < 0
            or bootstrap.get("confidence") != 0.95
            or bootstrap.get("method") != "paired_patient_cluster_percentile_image_weighted"
            or bootstrap.get("multiplicity") != "exploratory_pointwise_intervals_no_significance_claims"):
        raise ValueError("Use the bootstrap policy in configs/analysis.json.")
    seeds = config.get("training_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Expected unique nonnegative training seeds.")
    root, out = project_path(evaluation_root), project_path(output_root)
    if out.exists() or out.is_symlink():
        raise FileExistsError("Statistics directory exists; select a new output directory.")
    checkpoint_root = Path(checkpoint_root).expanduser().resolve()
    checkpoint_manifest_hash = sha256(checkpoint_root / "manifest.json")
    source_identity = None
    contrasts, per_seed = [], []
    for tissue in TISSUES:
        expected, manifest = read_manifest({"manifest_root": manifest_root}, tissue, "test", check_files=False)
        patients = [r["pid"] for r in expected]
        matrices, input_identity = {}, None
        for condition in CONDITIONS:
            values = []
            for training_seed in seeds:
                pq, summary = load_evaluation(root, tissue, condition, training_seed, expected,
                                              sha256(manifest), checkpoint_kind=CHECKPOINT_KIND)
                entry = _manifest_entry(checkpoint_root, tissue, condition, training_seed)
                if (summary.get("checkpoint_manifest_sha256") != checkpoint_manifest_hash
                        or summary.get("checkpoint_sha256") != entry["sha256"]
                        or summary.get("selected_update") != entry["selected_update"]
                        or any(summary.get("public_model", {}).get(k) != v for k, v in entry.items())):
                    raise ValueError("Evaluation model differs from the release manifest.")
                sources = summary.get("evaluation_sources")
                if not isinstance(sources, dict) or not sources:
                    raise ValueError("Evaluation source identity is missing.")
                if source_identity is None:
                    source_identity = sources
                elif sources != source_identity:
                    raise ValueError("Evaluation source versions differ.")
                inputs_path = root / tissue / condition / f"seed_{training_seed}/private/inputs.json"
                if sha256(inputs_path) != summary.get("input_records_sha256"):
                    raise ValueError("Evaluation input records changed.")
                inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
                if [(r["id"], r["pid"]) for r in inputs] != [(r["id"], r["pid"]) for r in expected]:
                    raise ValueError("Evaluation input identity/order differs from the manifest.")
                if input_identity is None:
                    input_identity = inputs
                elif inputs != input_identity:
                    raise ValueError("Evaluations used different image or mask inputs.")
                values.append(pq)
                per_seed.append({"tissue": tissue, "condition": condition, "seed": training_seed,
                                 "mean_pq": float(pq.mean())})
            matrices[condition] = np.stack(values)
        for a, b in CONTRASTS:
            result = paired_patient_bootstrap(matrices[a] - matrices[b], patients, repetitions, seed)
            result.update(tissue=tissue, contrast=a + "-" + b, seed_values=seeds)
            contrasts.append(result)
    out.mkdir(parents=True, exist_ok=False)
    report = {"checkpoint_manifest_sha256": checkpoint_manifest_hash,
              "analysis_config_sha256": sha256(project_path(config_path)),
              "evaluation_sources": source_identity, "primary_metric": "image_mean_pq",
              "contrasts": contrasts, "per_seed": per_seed,
              "scope": "Public-weight evaluations; original training records and historical split identity are not verified",
              "interpretation": "Exploratory pointwise 95% patient-sampling intervals conditional on completed seeds; "
                                "no multiplicity correction, significance, equivalence, or noninferiority claims.",
              "study_limitations": "The study test set was used in an earlier design review; "
                                   "generator-training/evaluation patient overlap has not been verified."}
    write_json(out / "statistics.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", default="data/manifests")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--config", default="configs/analysis.json")
    parser.add_argument("--evaluation-dir", default="outputs/public_evaluation")
    parser.add_argument("--output-dir", default="outputs/public_statistics")
    args = parser.parse_args(argv)
    report = summarize_public(args.evaluation_dir, args.manifest_root, args.checkpoint_dir,
                              args.config, args.output_dir)
    print(f"Summarized {len(report['contrasts'])} paired tissue-specific contrasts.")


if __name__ == "__main__":
    main()
