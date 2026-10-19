from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os

ROOT = Path(__file__).resolve().parents[1]
TISSUES = ("STNT", "STIN", "STDI", "STMX")
CONDITIONS = ("base", "add", "replace50", "replace100", "real300")
COUNTS = {"base": (600, 0), "add": (600, 600), "replace50": (300, 300),
          "replace100": (0, 600), "real300": (300, 0)}
POLICY = {
    "budget": "optimizer_updates", "early_stopping": False,
    "checkpoint_selection": "validation_image_mean_pq",
    "checkpoint_ties": "earliest_update", "primary_checkpoint": "best_pq.pth",
    "primary_metric": "image_mean_pq", "np_threshold": 0.5,
    "np_comparison": ">=", "match_iou": 0.5, "matching_comparison": ">",
    "empty_both": "all_scores_zero", "hv_channels": ["X", "Y"],
    "loss_weights": {"np_ce": 1.0, "np_dice": 1.0, "hv_mse": 1.0, "hv_msge": 1.0},
    "patches_per_image": 4, "model_mode": "fast", "drop_last": True,
    "frozen_encoder_bn": "eval", "test_used_in_design": True,
}


def execution_policy(config):
    return {**POLICY, "early_stopping": (config.get("lr_schedule") or {}).get("kind") == "plateau_once"}


def project_path(value, *, must_exist=False):
    candidate = (ROOT / value).resolve()
    if not candidate.is_relative_to(ROOT.resolve()):
        raise ValueError("Study input and output paths must stay inside the project.")
    if must_exist and not candidate.exists():
        raise FileNotFoundError("Required project file is missing.")
    return candidate


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = project_path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def source_hashes():
    paths = []
    for folder in ("src", "source_core"):
        paths.extend((ROOT / folder).rglob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256(project_path(p)) for p in sorted(paths)}


def load_config(path="configs/study.json"):
    config = json.loads(project_path(path, must_exist=True).read_text(encoding="utf-8-sig"))
    if list(config["tissues"]) != list(TISSUES) or config["conditions"] != list(CONDITIONS):
        raise ValueError("Configuration must match the four tissues and five conditions.")
    if config.get("policy") != execution_policy(config):
        raise ValueError("Study policy does not match the code. Use the matching policy file.")
    if config.get("model_mode") != "fast" or config.get("nuclear_type_classification") is not False:
        raise ValueError("This study requires fast-mode NP/HV instance segmentation.")
    project_path(config["manifest_root"], must_exist=True)
    project_path(config["pretrained"], must_exist=True)
    return config


def training_settings(config):
    required = ("training_seeds", "batch_size", "max_optimizer_updates", "lr_schedule",
                "validation_interval_updates", "encoder_freeze_schedule")
    missing = [key for key in required if config.get(key) is None]
    if missing:
        raise ValueError("Set the required training values: " + ", ".join(missing))
    settings = {key: config[key] for key in required}
    positive = ("batch_size", "max_optimizer_updates", "validation_interval_updates")
    for key in positive:
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(key + " must be a positive integer.")
    if settings["batch_size"] > min(sum(v) for v in COUNTS.values()) * POLICY["patches_per_image"]:
        raise ValueError("Batch size exceeds the total patches in the smallest condition.")
    seeds = settings["training_seeds"]
    if (not isinstance(seeds, list) or not seeds or
            any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or
            len(set(seeds)) != len(seeds)):
        raise ValueError("training_seeds must be a nonempty list of unique integers in [0, 2**32).")
    lr = settings["lr_schedule"]
    if not isinstance(lr, dict) or lr.get("kind") not in ("step", "plateau_once"):
        raise ValueError("lr_schedule.kind must be step or plateau_once.")
    for key in (("initial_lr", "gamma") if lr["kind"] == "step" else ("initial_lr", "reduced_lr")):
        if (type(lr.get(key)) not in (int, float) or not math.isfinite(lr[key]) or lr[key] <= 0):
            raise ValueError("LR setting must be finite and positive: " + key)
    if config.get("policy") != execution_policy(config):
        raise ValueError("LR or early-stopping settings do not match the policy.")
    if lr["kind"] == "step":
        if type(lr.get("step_size_updates")) is not int or lr["step_size_updates"] < 1:
            raise ValueError("LR reduction interval must be a positive integer number of updates.")
        if lr["gamma"] > 1:
            raise ValueError("LR gamma must not exceed 1.")
        if config.get("early_stopping") is not None:
            raise ValueError("Fixed StepLR pilots must not configure early stopping.")
    else:
        early = config.get("early_stopping")
        if not isinstance(early, dict):
            raise ValueError("plateau_once requires early_stopping settings.")
        if (type(early.get("min_delta")) not in (int, float) or
                not math.isfinite(early["min_delta"]) or not 0 <= early["min_delta"] < 1):
            raise ValueError("min_delta must be a finite absolute PQ difference in [0, 1).")
        if (type(lr.get("patience_checks")) is not int or lr["patience_checks"] < 1 or
                type(early.get("patience_checks")) is not int or
                early["patience_checks"] <= lr["patience_checks"]):
            raise ValueError("Early-stopping patience must exceed LR reduction patience.")
        start = early.get("start_after_updates")
        if (type(start) is not int or not 0 <= start < settings["max_optimizer_updates"] or
                start % settings["validation_interval_updates"]):
            raise ValueError("Patience counting must start at a validation interval multiple below the maximum update budget.")
        if lr["reduced_lr"] >= lr["initial_lr"]:
            raise ValueError("Reduced LR must be lower than initial LR.")
        settings["early_stopping"] = dict(early)
    freeze = settings["encoder_freeze_schedule"]
    if not isinstance(freeze, dict) or type(freeze.get("freeze_updates")) is not int:
        raise ValueError("encoder_freeze_schedule.freeze_updates must be an integer.")
    if not 0 <= freeze["freeze_updates"] <= settings["max_optimizer_updates"]:
        raise ValueError("Encoder freeze updates are out of range.")
    if settings["validation_interval_updates"] > settings["max_optimizer_updates"]:
        raise ValueError("Validation interval exceeds the total update budget.")
    return settings


def read_manifest(config, tissue, split, condition=None, *, check_files=True, checked_paths=None):
    if tissue not in TISSUES or split not in ("train", "val", "test"):
        raise ValueError("Unknown tissue or split.")
    folder = Path(config["manifest_root"]) / tissue
    if split == "train":
        if condition not in CONDITIONS:
            raise ValueError("A training condition is required.")
        folder /= condition
    path = project_path(folder / (split + ".csv"), must_exist=True)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "pid", "split", "source", "cancer_type", "image_path", "inst_mask_path"}
    if not rows or not required <= rows[0].keys():
        raise ValueError("Manifest is empty or missing required columns.")
    ids = set()
    for number, row in enumerate(rows, 1):
        if not row["id"] or row["id"] in ids:
            raise ValueError(f"Manifest row {number}: missing or duplicate image key.")
        ids.add(row["id"])
        if row["split"] != split or row["cancer_type"] != tissue:
            raise ValueError(f"Manifest row {number}: split or tissue mismatch.")
        if row["source"] not in ("real", "synthetic"):
            raise ValueError(f"Manifest row {number}: invalid data source.")
        if row["source"] == "real" and not row["pid"]:
            raise ValueError(f"Manifest row {number}: missing patient key for a real image.")
        for key in ("image_path", "inst_mask_path", "label_path", "class_mask_path"):
            if key in row:
                if not row[key]:
                    raise ValueError(f"Manifest row {number}: empty input path.")
                if checked_paths is None or row[key] not in checked_paths:
                    project_path(row[key], must_exist=check_files)
                    if checked_paths is not None:
                        checked_paths.add(row[key])
    counts = (sum(r["source"] == "real" for r in rows), sum(r["source"] == "synthetic" for r in rows))
    expected = COUNTS[condition] if split == "train" else (75, 0)
    if counts != expected:
        raise ValueError("Manifest real/synthetic counts do not match the specified condition.")
    return rows, path


def audit_manifests(config):
    patients = {s: set() for s in ("train", "val", "test")}
    manifest_hashes, counts, checked_paths = {}, {}, set()
    for tissue in TISSUES:
        train = {}
        for condition in CONDITIONS:
            rows, path = read_manifest(config, tissue, "train", condition, checked_paths=checked_paths)
            train[condition] = rows
            manifest_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
            counts[f"{tissue}/{condition}"] = len(rows)
            patients["train"].update(r["pid"] for r in rows if r["source"] == "real")
        if train["real300"] != [r for r in train["replace50"] if r["source"] == "real"]:
            raise ValueError("Real300 and Replace50 have different real-image rows.")
        def subset(condition, source):
            return {r["id"]: r for r in train[condition] if r["source"] == source}
        if subset("base", "real") != subset("add", "real"):
            raise ValueError("Base and Add have different real-image sets.")
        if subset("replace100", "synthetic") != subset("add", "synthetic"):
            raise ValueError("Replace100 and Add have different synthetic-image sets.")
        for source in ("real", "synthetic"):
            full = subset("add", source)
            if any(full.get(k) != v for k, v in subset("replace50", source).items()):
                raise ValueError("Replace50 does not match the fixed subset of Add.")
        for split in ("val", "test"):
            rows, path = read_manifest(config, tissue, split, checked_paths=checked_paths)
            patients[split].update(r["pid"] for r in rows)
            manifest_hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
    overlaps = {a + "_" + b: len(patients[a] & patients[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()):
        raise ValueError("Patient overlap between splits was found across tissues.")
    return {"training_image_counts": counts, "patient_overlap": overlaps,
            "unique_data_paths_checked": len(checked_paths),
            "patients": {s: len(p) for s, p in patients.items()}, "manifest_sha256": manifest_hashes}


def main():
    parser = argparse.ArgumentParser(description="Validate study settings and fixed data splits.")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--inventory", action="store_true", help="Audit fixed splits without requiring training values")
    args = parser.parse_args()
    config = load_config(args.config)
    result = audit_manifests(config)
    result.pop("manifest_sha256")
    if not args.inventory:
        settings = training_settings(config)
        result["planned_runs"] = 20 * len(settings["training_seeds"])
        result["training_settings"] = settings
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
