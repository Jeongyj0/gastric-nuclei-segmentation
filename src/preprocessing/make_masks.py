from pathlib import Path, PureWindowsPath
from contextlib import ExitStack
import argparse
import csv
import json

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = (
    "id", "source", "split", "cancer_type", "image_path", "label_path",
    "inst_mask_path", "class_mask_path",
)
CLASS_TO_ID = {"Stroma": 1, "Immune": 2, "Normal": 3, "Tumor": 4}
METADATA_VALUES = {
    "source": ("real", "synthetic"), "split": ("train", "val", "test"),
    "cancer_type": ("STNT", "STIN", "STDI", "STMX"),
}


def get_label(obj):
    return obj.get("label") or obj.get("label_nm")


def get_polygon(obj):
    coords = obj.get("coordinate")

    if not isinstance(coords, list) or len(coords) < 3:
        return None

    polygon = []

    for p in coords:
        if isinstance(p, dict):
            x, y = p["x"], p["y"]
        else:
            x, y = p[0], p[1]

        polygon.append(
            (int(round(float(x))), int(round(float(y))))
        )

    return polygon


def json_to_masks(json_path, height, width):
    with json_path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    objects = data["content"]["file"]["object"]

    inst_mask = np.zeros((height, width), dtype=np.int32)
    class_mask = np.zeros((height, width), dtype=np.uint8)

    instance_id = 1

    for obj in objects:
        label = get_label(obj)

        if label not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown label '{label}' in {json_path}"
            )

        polygon = get_polygon(obj)

        if polygon is None:
            raise ValueError(
                f"Invalid polygon in {json_path}"
            )

        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=1)
        region = np.asarray(mask, dtype=bool) & (inst_mask == 0)

        if not region.any():
            continue

        inst_mask[region] = instance_id
        class_mask[region] = CLASS_TO_ID[label]
        instance_id += 1

    return inst_mask, class_mask


def _project_path(root, value):
    if Path(value).is_absolute() or PureWindowsPath(value).drive or ":" in str(value):
        raise ValueError("Metadata and file paths must be relative to --root.")
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Input and output paths must stay inside --root.")
    return path


def _read_metadata(path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if (not fields or len(fields) != len(set(fields))
                or any(key not in fields for key in REQUIRED_COLUMNS)):
            raise ValueError("Metadata requires unique columns: " + ", ".join(REQUIRED_COLUMNS))
        rows = list(reader)
    if not rows:
        raise ValueError("Metadata contains no image rows.")
    ids = set()
    for index, row in enumerate(rows, 1):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Metadata row {index} does not match the column count.")
        if any(not row[key].strip() for key in REQUIRED_COLUMNS):
            raise ValueError(f"Metadata row {index} has an empty required field.")
        if row["id"] in ids:
            raise ValueError("Metadata image IDs must be unique.")
        ids.add(row["id"])
        for key, allowed in METADATA_VALUES.items():
            if row[key] not in allowed:
                raise ValueError(f"Metadata row {index} has an unsupported {key} value.")
    return fields, rows


def _check_existing_pair(paths, expected):
    if not all(path.is_file() for path in paths):
        raise FileExistsError("Existing masks must form a complete pair of files.")
    for path, mask in zip(paths, expected):
        actual = np.load(path, allow_pickle=False)
        if actual.dtype != mask.dtype or actual.shape != mask.shape or not np.array_equal(actual, mask):
            raise ValueError("Existing masks differ from the JSON-derived masks; no files were replaced.")


def _save_pair(paths, masks):
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                streams.append(stack.enter_context(path.open("xb")))
                created.append(path)
            for stream, mask in zip(streams, masks):
                np.save(stream, mask, allow_pickle=False)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build instance and class masks from polygon JSON labels.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--meta", type=Path, default=Path("data/selected_data.csv"))
    parser.add_argument("--output-meta", type=Path, default=Path("data/meta.csv"))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("--root must be an existing project directory.")
    meta = _project_path(root, args.meta)
    output_meta = _project_path(root, args.output_meta)
    fields, rows = _read_metadata(meta)
    inputs = {meta}
    outputs = {output_meta}
    prepared = []
    for row in rows:
        paths = {key: _project_path(root, row[key]) for key in
                 ("image_path", "label_path", "inst_mask_path", "class_mask_path")}
        for key in ("image_path", "label_path"):
            if not paths[key].is_file():
                raise FileNotFoundError(f"Required {key} file is missing.")
            inputs.add(paths[key])
        pair = (paths["inst_mask_path"], paths["class_mask_path"])
        for path in pair:
            if path in outputs:
                raise ValueError("Each mask and output metadata file must have a distinct path.")
            outputs.add(path)
        present = [path.exists() or path.is_symlink() for path in pair]
        if any(present):
            if not args.reuse_existing:
                raise FileExistsError("Mask outputs exist; use --reuse-existing only for matching complete pairs.")
            if not all(present) or not all(path.is_file() for path in pair):
                raise FileExistsError("Existing masks must form a complete pair of files.")
        prepared.append((paths, pair))
    if inputs & outputs:
        raise ValueError("Mask and metadata outputs must not replace input files.")
    if output_meta.exists() or output_meta.is_symlink():
        if not args.reuse_existing:
            raise FileExistsError("Output metadata exists; use --reuse-existing only for matching results.")
        if not output_meta.is_file() or _read_metadata(output_meta) != (fields, rows):
            raise ValueError("Existing output metadata differs from the selected metadata.")
    created = reused = 0
    for paths, pair in prepared:
        with Image.open(paths["image_path"]) as image:
            width, height = image.size
        masks = json_to_masks(paths["label_path"], height, width)
        if any(path.exists() or path.is_symlink() for path in pair):
            if not args.reuse_existing:
                raise FileExistsError("Mask outputs appeared during processing; no existing files were replaced.")
            _check_existing_pair(pair, masks)
            reused += 1
        else:
            _save_pair(pair, masks)
            created += 1
    if output_meta.exists() or output_meta.is_symlink():
        if not args.reuse_existing:
            raise FileExistsError("Output metadata appeared during processing; no existing files were replaced.")
        if not output_meta.is_file() or _read_metadata(output_meta) != (fields, rows):
            raise ValueError("Existing output metadata differs from the selected metadata.")
    else:
        output_meta.parent.mkdir(parents=True, exist_ok=True)
        stream = output_meta.open("x", encoding="utf-8", newline="")
        try:
            with stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        except BaseException:
            output_meta.unlink(missing_ok=True)
            raise
    print(json.dumps({"processed": len(rows), "created": created, "reused": reused,
                      "output_meta": output_meta.relative_to(root).as_posix()}))


if __name__ == "__main__":
    main()
