from pathlib import Path, PurePosixPath, PureWindowsPath
from contextlib import ExitStack
import argparse
import hashlib
import json
import math
import sys


ROOT = Path(__file__).resolve().parents[2]
TISSUES = ("STNT", "STIN", "STDI", "STMX")
CONDITIONS = ("base", "real300", "add", "replace50", "replace100")
ARCHITECTURE = {
    "name": "hovernet", "mode": "fast", "input_channels": 3,
    "nuclear_type_classification": False, "outputs": ["np", "hv"],
}
MODEL_FIELDS = (
    "tissue", "condition", "seed", "selected_update", "file", "sha256",
    "bytes", "tensor_count", "tensor_bytes", "source_checkpoint_sha256",
)


def _checkpoint_path(root, filename):
    if (not isinstance(filename, str) or not filename or "\\" in filename
            or PurePosixPath(filename).is_absolute() or PureWindowsPath(filename).drive
            or ":" in filename or any(part in ("", ".", "..") for part in filename.split("/"))):
        raise ValueError("Checkpoint file must be a relative path inside the checkpoint directory.")
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Checkpoint file resolves outside the checkpoint directory.")
    return path


def _manifest_entry(root, tissue, condition, seed):
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Checkpoint manifest.json is missing; provide --checkpoint-dir.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or type(manifest.get("format_version")) is not int
            or manifest["format_version"] != 1):
        raise ValueError("Unsupported checkpoint manifest format.")
    architecture = manifest.get("architecture")
    if (architecture != ARCHITECTURE or type(architecture.get("input_channels")) is not int
            or architecture.get("nuclear_type_classification") is not False):
        raise ValueError("Checkpoints must use fast HoVerNet with RGB input and NP/HV outputs.")
    entries = manifest.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Checkpoint manifest must contain model entries.")
    identities = set()
    selected = None
    for entry in entries:
        if not isinstance(entry, dict) or any(key not in entry for key in MODEL_FIELDS):
            raise ValueError("Checkpoint model entry is incomplete.")
        if (entry["tissue"] not in TISSUES or entry["condition"] not in CONDITIONS
                or type(entry["seed"]) is not int or entry["seed"] < 0):
            raise ValueError("Invalid checkpoint model identity.")
        identity = (entry["tissue"], entry["condition"], entry["seed"])
        if identity in identities:
            raise ValueError("Duplicate model identity in checkpoint manifest.")
        identities.add(identity)
        for key in ("selected_update", "bytes", "tensor_count", "tensor_bytes"):
            if type(entry[key]) is not int or entry[key] < 1:
                raise ValueError(f"Checkpoint {key} must be a positive integer.")
        for key in ("sha256", "source_checkpoint_sha256"):
            value = entry[key]
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in "0123456789abcdef" for char in value.lower())):
                raise ValueError(f"Invalid checkpoint {key}.")
        _checkpoint_path(root, entry["file"])
        if identity == (tissue, condition, seed):
            selected = {key: entry[key] for key in MODEL_FIELDS}
    if selected is None:
        raise ValueError(f"No public checkpoint for {tissue}/{condition}/seed_{seed}.")
    return selected


def _create_model():
    core = str(ROOT / "source_core" / "hovernet")
    if core not in sys.path:
        sys.path.insert(0, core)
    from models.hovernet.net_desc import create_model
    return create_model(input_ch=3, nr_types=None, freeze=False, mode="fast")


def load_public_model(tissue, condition, seed=42, *, device="cpu", checkpoint_root=None):
    if (tissue not in TISSUES or condition not in CONDITIONS
            or type(seed) is not int or seed < 0):
        raise ValueError("Specify a valid tissue, condition, and nonnegative integer seed.")
    root = (Path(checkpoint_root) if checkpoint_root is not None else ROOT / "checkpoints").expanduser().resolve()
    entry = _manifest_entry(root, tissue, condition, seed)
    path = _checkpoint_path(root, entry["file"])
    if not path.is_file():
        raise FileNotFoundError(f"Public checkpoint is missing: {entry['file']}.")
    import torch
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Device must be cpu or cuda.")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")
    with path.open("rb") as stream:
        stream.seek(0, 2)
        if stream.tell() != entry["bytes"]:
            stream.seek(0)
            if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Checkpoint is a Git LFS pointer. Download and extract the separate Zenodo weight archive.")
            raise ValueError("Checkpoint file size does not match manifest.")
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != entry["sha256"].lower():
            raise ValueError("Checkpoint SHA-256 does not match manifest.")
        stream.seek(0)
        state = torch.load(stream, weights_only=True, map_location="cpu")
    if (not isinstance(state, dict) or not state
            or any(not isinstance(key, str) or not isinstance(value, torch.Tensor)
                   for key, value in state.items())):
        raise ValueError("Public checkpoint must be a tensor-only state dictionary.")
    if (len(state) != entry["tensor_count"]
            or sum(tensor.numel() * tensor.element_size() for tensor in state.values()) != entry["tensor_bytes"]):
        raise ValueError("Checkpoint tensor count or byte size does not match manifest.")
    model = _create_model()
    expected_state = model.state_dict()
    if any(tensor.dtype != expected_state[key].dtype for key, tensor in state.items() if key in expected_state):
        raise ValueError("Checkpoint tensor dtypes do not match the model.")
    model.load_state_dict(state, strict=True)
    model = model.to(target).eval()
    metadata = {**entry, "format_version": 1, "architecture": {**ARCHITECTURE, "outputs": ["np", "hv"]}}
    return model, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description="Segment nuclei using a public HoVerNet checkpoint.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--tissue", choices=TISSUES, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prediction"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    out = args.output_dir.expanduser().resolve()
    names = ("instances.npy", "raw_maps.npy", "prediction.json")
    if out.exists() and not out.is_dir():
        raise NotADirectoryError("Output directory is an existing file.")
    if not args.overwrite and any((out / name).exists() or (out / name).is_symlink() for name in names):
        raise FileExistsError("Prediction outputs already exist; use --overwrite or another output directory.")
    import numpy as np
    from PIL import Image
    from src.eval.inference import predict_raw_map, postprocess
    with Image.open(args.image.expanduser()) as image:
        image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    model, metadata = load_public_model(args.tissue, args.condition, args.seed,
        device=args.device, checkpoint_root=args.checkpoint_dir)
    raw, patches = predict_raw_map(model, image, args.device, batch_size=args.batch_size)
    if raw.shape != (*image.shape[:2], 3) or not np.isfinite(raw).all():
        raise ValueError("Prediction must contain finite NP/HV maps matching the image size.")
    instances, _ = postprocess(raw, return_centroids=False)
    if (instances.shape != image.shape[:2] or not np.issubdtype(instances.dtype, np.integer)
            or np.any(instances < 0)):
        raise ValueError("Postprocessing must return nonnegative integer instance labels.")
    report = {
        "model": metadata,
        "image": {"height": image.shape[0], "width": image.shape[1], "channels": 3},
        "instance_count": int(np.count_nonzero(np.unique(instances) > 0)),
        "device": args.device, "batch_size": args.batch_size,
        "patch_count": patches, "batch_count": math.ceil(patches / args.batch_size),
        "raw_map_channels": ["NP", "H", "V"],
        "outputs": {"instances": names[0], "raw_maps": names[1]},
    }
    out.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for name in names:
                path = out / name
                mode = "w" if args.overwrite else "x"
                options = {"encoding": "utf-8"} if name.endswith(".json") else {}
                stream = stack.enter_context(path.open(mode if options else mode + "b", **options))
                streams.append(stream)
                if not args.overwrite:
                    created.append(path)
            np.save(streams[0], instances, allow_pickle=False)
            np.save(streams[1], raw, allow_pickle=False)
            streams[2].write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except BaseException:
        for path in created:
            try:
                if path.stat().st_size == 0:
                    path.unlink()
            except OSError:
                pass
        raise
    print(json.dumps({key: report[key] for key in ("instance_count", "patch_count", "batch_count")}))


if __name__ == "__main__":
    main()
