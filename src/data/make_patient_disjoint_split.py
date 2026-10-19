from pathlib import Path, PureWindowsPath
import argparse
import json
import math

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[2]
CANCERS = ["STNT", "STIN", "STDI", "STMX"]
SPLITS = ["train", "val", "test"]
TARGET = {"train": 600, "val": 75, "test": 75}
SEED = 42
PATH_COLUMNS = ("image_path", "label_path", "inst_mask_path", "class_mask_path")


def project_path(root, value, *, must_exist=False):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Input and output paths must be nonempty project-relative paths.")
    value = str(value)
    if Path(value).is_absolute() or PureWindowsPath(value).drive or ":" in value:
        raise ValueError("Input and output paths must be project-relative.")
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Input and output paths must resolve inside the project root.")
    if must_exist and not path.is_file():
        raise FileNotFoundError("A required project input file is missing.")
    return path


def valid_pid(value):
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().casefold() not in {"nan", "none", "null", "na", "n/a"}
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        return math.isfinite(value)
    return False


def validate_table(meta, *, require_pid=False, require_label=False):
    required = {"id", "source", "split", "cancer_type", "image_path", "inst_mask_path"}
    if require_pid:
        required.add("pid")
    if require_label:
        required.add("label_path")
    if meta.empty or not required.issubset(meta.columns) or meta.columns.duplicated().any():
        raise ValueError("Metadata is empty or missing required unique columns.")
    if meta["id"].isna().any() or meta["id"].astype(str).str.strip().eq("").any() or meta["id"].duplicated().any():
        raise ValueError("Image IDs must be nonempty and unique across metadata.")
    for column, allowed in (("source", ("real", "synthetic")), ("split", SPLITS), ("cancer_type", CANCERS)):
        if not meta[column].isin(allowed).all():
            raise ValueError(f"Invalid {column} in metadata.")
    if require_pid:
        real = meta[meta["source"] == "real"]
        if not real["pid"].map(valid_pid).all():
            raise ValueError("Missing or invalid PID in real data.")


def read_meta(path, root, *, require_pid=False, require_label=False):
    meta = pd.read_csv(path, dtype={"id": str, "pid": str})
    validate_table(meta, require_pid=require_pid, require_label=require_label)
    for column in PATH_COLUMNS:
        if column in meta:
            for value in meta[column].drop_duplicates():
                project_path(root, value, must_exist=True)
    return meta


def load_pid(label_path):
    with label_path.open("r", encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("content"), dict):
        raise ValueError("Real annotation must contain content.clinical.pid.")
    clinical = data["content"].get("clinical")
    if not isinstance(clinical, dict) or not valid_pid(clinical.get("pid")):
        raise ValueError("Real annotation must contain a valid content.clinical.pid.")
    pid = clinical["pid"]
    if isinstance(pid, str):
        return pid.strip()
    if isinstance(pid, (float, np.floating)) and not pid.is_integer():
        return str(pid)
    return str(int(pid))


def assign_patients(real):
    groups = real.groupby(["cancer_type", "pid"]).size().reset_index(name="images")
    keys = list(groups[["cancer_type", "pid"]].itertuples(index=False, name=None))
    index = {}
    n = 0
    for cancer, pid in keys:
        for split in SPLITS:
            index[(cancer, pid, split)] = n
            n += 1
    rows = []
    cols = []
    values = []
    bounds = []
    constraint = 0
    for cancer, pid in keys:
        for split in SPLITS:
            rows.append(constraint)
            cols.append(index[(cancer, pid, split)])
            values.append(1.0)
        bounds.append(1.0)
        constraint += 1
    count_map = {(row.cancer_type, row.pid): row.images for row in groups.itertuples()}
    for cancer in CANCERS:
        cancer_pids = [pid for c, pid in keys if c == cancer]
        for split in SPLITS:
            for pid in cancer_pids:
                rows.append(constraint)
                cols.append(index[(cancer, pid, split)])
                values.append(float(count_map[(cancer, pid)]))
            bounds.append(float(TARGET[split]))
            constraint += 1
    pid_groups = groups.groupby("pid")["cancer_type"].apply(list).to_dict()
    for pid, cancers in pid_groups.items():
        if len(cancers) < 2:
            continue
        first = cancers[0]
        for other in cancers[1:]:
            for split in SPLITS:
                rows.extend([constraint, constraint])
                cols.extend([index[(first, pid, split)], index[(other, pid, split)]])
                values.extend([1.0, -1.0])
                bounds.append(0.0)
                constraint += 1
    matrix = coo_matrix((values, (rows, cols)), shape=(constraint, n)).tocsr()
    bounds_array = np.asarray(bounds)
    rng = np.random.default_rng(SEED)
    objective = rng.random(n) * 1e-6
    result = milp(c=objective, integrality=np.ones(n), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, bounds_array, bounds_array))
    if not result.success:
        raise RuntimeError("Patient allocation is infeasible or the solver did not finish: " + str(result.message))
    assignments = []
    for cancer, pid in keys:
        scores = {split: result.x[index[(cancer, pid, split)]] for split in SPLITS}
        split = max(scores, key=scores.get)
        assignments.append({"cancer_type": cancer, "pid": pid, "split": split,
                            "images": count_map[(cancer, pid)]})
    return pd.DataFrame(assignments)


def make_patient_disjoint_meta(meta, root):
    validate_table(meta, require_label=True)
    real = meta[meta["source"] == "real"].copy()
    if real.empty:
        raise ValueError("Real images are required for patient allocation.")
    real["pid"] = [load_pid(project_path(root, path, must_exist=True)) for path in real["label_path"]]
    assignment = assign_patients(real)
    real = real.merge(assignment[["cancer_type", "pid", "split"]],
                      on=["cancer_type", "pid"], suffixes=("_original", "_new"))
    split_map = dict(zip(real["id"], real["split_new"]))
    pid_map = dict(zip(real["id"], real["pid"]))
    new_meta = meta.copy()
    new_meta["original_split"] = new_meta["split"]
    new_meta["pid"] = pd.NA
    mask = new_meta["source"] == "real"
    new_meta.loc[mask, "split"] = new_meta.loc[mask, "id"].map(split_map)
    new_meta.loc[mask, "pid"] = new_meta.loc[mask, "id"].map(pid_map)
    validate_table(new_meta, require_pid=True)
    real_meta = new_meta[new_meta["source"] == "real"]
    if real_meta.groupby("pid")["split"].nunique().gt(1).any():
        raise ValueError("Patient split overlap detected after allocation.")
    for cancer in CANCERS:
        counts = real_meta[real_meta["cancer_type"] == cancer]["split"].value_counts()
        if any(counts.get(split, 0) != count for split, count in TARGET.items()):
            raise ValueError("Patient allocation did not meet the required image counts.")
    return new_meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Allocate real patients across tissues to disjoint study splits.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--meta", default="data/meta.csv")
    parser.add_argument("--output", default="data/meta_patient_disjoint.csv")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("Project root must be a directory.")
    source = project_path(root, args.meta, must_exist=True)
    output = project_path(root, args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Patient metadata output already exists; choose another output path.")
    meta = read_meta(source, root, require_label=True)
    result = make_patient_disjoint_meta(meta, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open("x", encoding="utf-8-sig", newline="")
    try:
        with stream:
            result.to_csv(stream, index=False)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print("Patient allocation complete: 600 train, 75 validation, and 75 test images per tissue; no patient overlap.")


if __name__ == "__main__":
    main()
