from pathlib import Path
import argparse
import shutil

import numpy as np
import pandas as pd

from src.data.make_patient_disjoint_split import CANCERS, ROOT, SEED, TARGET, project_path, read_meta, validate_table


CONDITIONS = {"base": (600, 0), "add": (600, 600), "replace50": (300, 300),
              "replace100": (0, 600), "real300": (300, 0)}


def shuffled(df):
    df = df.sort_values("id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    return df.iloc[order].reset_index(drop=True)


def validate_meta(meta):
    validate_table(meta, require_pid=True)
    real = meta[meta["source"] == "real"]
    if (real.groupby("pid")["split"].nunique() > 1).any():
        raise ValueError("Patient split overlap detected.")
    for cancer in CANCERS:
        cancer_real = real[real["cancer_type"] == cancer]
        counts = cancer_real["split"].value_counts()
        for split, count in TARGET.items():
            if counts.get(split, 0) != count:
                raise ValueError(f"{cancer} {split}: expected {count}, got {counts.get(split, 0)}.")
        synthetic = meta[(meta["source"] == "synthetic") & (meta["cancer_type"] == cancer)]
        if len(synthetic) != 600 or not synthetic["split"].eq("train").all():
            raise ValueError(f"{cancer}: expected exactly 600 synthetic training images.")


def build_condition(real, synthetic, real_n, syn_n):
    parts = []
    if real_n:
        parts.append(real.iloc[:real_n])
    if syn_n:
        parts.append(synthetic.iloc[:syn_n])
    train = pd.concat(parts, ignore_index=True)
    return train.sample(frac=1, random_state=SEED).reset_index(drop=True)


def validate_conditions(datasets):
    def rows(name, source):
        return datasets[name].loc[datasets[name]["source"] == source].sort_values("id").reset_index(drop=True)
    if not rows("base", "real").equals(rows("add", "real")):
        raise ValueError("Base and Add real-image rows differ.")
    if not rows("replace100", "synthetic").equals(rows("add", "synthetic")):
        raise ValueError("Add and Replace100 synthetic-image rows differ.")
    for source in ("real", "synthetic"):
        subset = rows("replace50", source)
        full = rows("add", source)
        if not subset.equals(full[full["id"].isin(subset["id"])].reset_index(drop=True)):
            raise ValueError("Replace50 is not a fixed subset of Add.")
    expected_real300 = datasets["replace50"].loc[datasets["replace50"]["source"] == "real"].reset_index(drop=True)
    if not datasets["real300"].reset_index(drop=True).equals(expected_real300):
        raise ValueError("Real300 must preserve the real-image rows and order of Replace50.")
    for name, (real_n, syn_n) in CONDITIONS.items():
        counts = datasets[name]["source"].value_counts()
        if (counts.get("real", 0), counts.get("synthetic", 0)) != (real_n, syn_n):
            raise ValueError("Condition image counts differ from the study specification.")


def build_manifests(meta):
    validate_meta(meta)
    manifests = {}
    for cancer in CANCERS:
        real = meta[(meta["source"] == "real") & (meta["cancer_type"] == cancer) & (meta["split"] == "train")]
        synthetic = meta[(meta["source"] == "synthetic") & (meta["cancer_type"] == cancer) & (meta["split"] == "train")]
        real = shuffled(real)
        synthetic = shuffled(synthetic)
        datasets = {name: build_condition(real, synthetic, real_n, syn_n)
                    for name, (real_n, syn_n) in CONDITIONS.items() if name != "real300"}
        datasets["real300"] = datasets["replace50"].loc[datasets["replace50"]["source"] == "real"].copy()
        validate_conditions(datasets)
        for name, train in datasets.items():
            manifests[f"{cancer}/{name}/train.csv"] = train
        for split in ("val", "test"):
            manifests[f"{cancer}/{split}.csv"] = meta[(meta["source"] == "real") & (meta["cancer_type"] == cancer)
                                                        & (meta["split"] == split)].sort_values("id")
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the five study conditions from patient-disjoint metadata.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--meta", default="data/meta_patient_disjoint.csv")
    parser.add_argument("--output-dir", default="data/manifests")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("Project root must be a directory.")
    source = project_path(root, args.meta, must_exist=True)
    output = project_path(root, args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Manifest output directory already exists; choose another directory.")
    meta = read_meta(source, root, require_pid=True)
    manifests = build_manifests(meta)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for relative, frame in manifests.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8-sig", newline="") as stream:
                frame.to_csv(stream, index=False)
    except BaseException:
        if output.resolve().is_relative_to(root) and output.resolve() != root:
            shutil.rmtree(output)
        raise
    print("Created 28 manifests for four tissues and five conditions with seed 42.")


if __name__ == "__main__":
    main()
