import argparse
import csv
import json
from pathlib import Path
import numpy as np
from src.study import (TISSUES, CONDITIONS, load_config, training_settings, read_manifest,
                       project_path, sha256, write_json)

CONTRASTS = (("add", "base"), ("replace50", "base"),
             ("replace50", "real300"), ("replace100", "base"))


def paired_patient_bootstrap(differences, patients, repetitions, seed, confidence=0.95):
    differences = np.asarray(differences, dtype=np.float64)
    if (differences.ndim != 2 or differences.shape[0] < 1 or differences.shape[1] < 1 or
            differences.shape[1] != len(patients) or not np.isfinite(differences).all()):
        raise ValueError("Invalid paired-difference matrix or patient count.")
    if type(repetitions) is not int or repetitions < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap repetitions or confidence level.")
    groups, inverse = np.unique(np.asarray(patients, dtype=str), return_inverse=True)
    means = differences.mean(axis=0)
    totals = np.bincount(inverse, weights=means)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    distribution = np.empty(repetitions, dtype=np.float64)
    for start in range(0, repetitions, 1000):
        stop = min(start + 1000, repetitions)
        draws = rng.integers(0, len(groups), size=(stop - start, len(groups)))
        distribution[start:stop] = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    alpha = (1 - confidence) / 2
    limits = np.quantile(distribution, [alpha, 1 - alpha])
    per_seed = differences.mean(axis=1)
    return {"mean_difference": float(means.mean()), "ci_low": float(limits[0]),
            "ci_high": float(limits[1]), "confidence": confidence,
            "training_seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
            "per_seed_differences": per_seed.tolist(), "images": len(means),
            "patients": len(groups), "training_seeds": len(per_seed),
            "repetitions": repetitions, "bootstrap_seed": seed,
            "ci_scope": "Evaluation-patient sampling uncertainty conditional on the completed seeds",
            "ci_method": "patient_cluster_percentile_image_weighted"}


def load_evaluation(root, tissue, condition, seed, expected_rows, manifest_hash, *, checkpoint_kind="best_pq.pth"):
    folder = project_path(root / tissue / condition / f"seed_{seed}", must_exist=True)
    status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "test_summary.json").read_text(encoding="utf-8"))
    if status["status"] != "completed" or summary["checkpoint"] != checkpoint_kind:
        raise ValueError("Statistics require completed evaluations of the expected checkpoint kind.")
    if (summary["tissue"], summary["condition"], summary["seed"]) != (tissue, condition, seed):
        raise ValueError("Evaluation identity mismatch.")
    if summary["test_manifest_sha256"] != manifest_hash:
        raise ValueError("Test manifests differ.")
    path = folder / "private/test_metrics.csv"
    if sha256(path) != summary["metrics_sha256"]:
        raise ValueError("Per-image evaluation hash mismatch.")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    pairs = [(r["id"], r["pid"]) for r in rows]
    expected = [(r["id"], r["pid"]) for r in expected_rows]
    if len(set(i for i, _ in pairs)) != len(pairs) or set(pairs) != set(expected):
        raise ValueError("Evaluation image/patient mapping or image count differs.")
    mapping = {r["id"]: float(r["pq"]) for r in rows}
    values = np.asarray([mapping[i] for i, _ in expected])
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid per-image PQ values.")
    if not np.isclose(values.mean(), summary["mean_pq"], rtol=0, atol=1e-12):
        raise ValueError("Per-image PQ does not match the evaluation summary.")
    return values, summary


def main():
    parser = argparse.ArgumentParser(description="Summarize paired PQ contrasts with patient bootstrap intervals.")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--evaluation-dir", default="outputs/evaluation")
    parser.add_argument("--output-dir", default="outputs/statistics")
    args = parser.parse_args()
    config = load_config(args.config)
    seeds = training_settings(config)["training_seeds"]
    bootstrap = config.get("bootstrap", {})
    repetitions, bootstrap_seed = bootstrap.get("repetitions"), bootstrap.get("seed")
    if type(repetitions) is not int or repetitions < 1 or type(bootstrap_seed) is not int or bootstrap_seed < 0:
        parser.error("Set bootstrap.repetitions and bootstrap.seed in the shared configuration.")
    if (bootstrap.get("confidence") != 0.95 or
            bootstrap.get("method") != "paired_patient_cluster_percentile_image_weighted" or
            bootstrap.get("multiplicity") != "exploratory_pointwise_intervals_no_significance_claims"):
        parser.error("Configured statistical reporting policy does not match the implementation.")
    root, out = project_path(args.evaluation_dir), project_path(args.output_dir)
    if out.exists():
        raise FileExistsError("Statistics directory already exists. Specify a new output directory.")
    contrasts, seed_scores, protocol_hashes = [], [], set()
    for tissue in TISSUES:
        expected, manifest = read_manifest(config, tissue, "test", check_files=False)
        patients = [r["pid"] for r in expected]
        matrices, initial_hashes = {}, {}
        for condition in CONDITIONS:
            values = []
            for seed in seeds:
                pq, summary = load_evaluation(root, tissue, condition, seed, expected, sha256(manifest))
                protocol_hashes.add(summary["protocol_sha256"])
                previous = initial_hashes.setdefault(seed, summary["initial_state_sha256"])
                if previous != summary["initial_state_sha256"]:
                    raise ValueError("Initial weights differ across conditions with the same seed.")
                values.append(pq)
                seed_scores.append({"tissue": tissue, "condition": condition, "seed": seed,
                                    "mean_pq": float(pq.mean())})
            matrices[condition] = np.stack(values)
        for a, b in CONTRASTS:
            result = paired_patient_bootstrap(matrices[a] - matrices[b], patients,
                                              repetitions, bootstrap_seed)
            result.update(tissue=tissue, contrast=a + "-" + b, seed_values=seeds)
            contrasts.append(result)
    if len(protocol_hashes) != 1:
        raise ValueError("Shared policy, code, or settings differ between runs.")
    condition_scores = []
    for tissue in TISSUES:
        for condition in CONDITIONS:
            values = [r["mean_pq"] for r in seed_scores
                      if r["tissue"] == tissue and r["condition"] == condition]
            condition_scores.append({"tissue": tissue, "condition": condition,
                                     "mean_pq_across_seeds": float(np.mean(values)),
                                     "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "statistics.json", {
        "protocol_sha256": next(iter(protocol_hashes)), "primary_metric": "image_mean_pq",
        "contrasts": contrasts, "per_seed": seed_scores, "condition_summary": condition_scores,
        "interpretation": "Exploratory tissue-specific estimates and individual 95% CIs; no multiplicity correction or significance claims. "
                          "Patient-sampling uncertainty and seed variability are separate; no equivalence or noninferiority claims.",
        "test_used_in_design": True, "generator_patient_overlap_verified": False})
    report = ["# Gastric nuclei instance segmentation statistics", "",
              "Primary metric: arithmetic mean of per-image PQ on the same real-only test set.", "",
              "| Tissue | Contrast | PQ difference | Patient bootstrap 95% CI | SD of seed differences |",
              "|---|---|---:|---:|---:|"]
    for r in contrasts:
        sd = "Unavailable (S=1)" if r["training_seed_sd"] is None else f"{r['training_seed_sd']:.5f}"
        report.append(f"| {r['tissue']} | {r['contrast']} | {r['mean_difference']:.5f} | "
                      f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}] | {sd} |")
    report.extend(["", "Differences were paired by image and seed, averaged over seeds, then patients were sampled with replacement. "
                   "All images of sampled patients were included, with the resampled image count as denominator. "
                   "CIs quantify evaluation-patient sampling uncertainty conditional on the completed seeds.", "",
                   "The 16 comparisons have individual exploratory CIs, without simultaneous coverage or multiplicity correction. "
                   "A CI containing zero does not establish equivalence or interchangeability.", "",
                   "The test set was used in an earlier design review. Overlap between generative-model training patients and downstream evaluation patients "
                   "has not been verified. Replace100 means no real images were used for downstream training."])
    (out / "statistics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Summarized {len(contrasts)} tissue-specific contrasts. Patient keys are excluded from the report.")


if __name__ == "__main__":
    main()
