from pathlib import Path
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.eval import predict


class PublicModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "checkpoints"
        self.path = self.root / "STNT/base/seed_42.pth"
        self.path.parent.mkdir(parents=True)
        self.save_state({"weight": torch.tensor([[1.0, 2.0]])})

    def save_state(self, state):
        torch.save(state, self.path)
        self.entry = {
            "tissue": "STNT", "condition": "base", "seed": 42,
            "selected_update": 1200, "file": "STNT/base/seed_42.pth",
            "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "bytes": self.path.stat().st_size, "tensor_count": len(state),
            "tensor_bytes": sum(value.numel() * value.element_size()
                                for value in state.values() if isinstance(value, torch.Tensor)),
            "source_checkpoint_sha256": "0" * 64,
        }
        self.manifest = {"format_version": 1, "architecture": dict(predict.ARCHITECTURE), "models": [self.entry]}
        self.write_manifest()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def load(self, **kwargs):
        with patch.object(predict, "_create_model", side_effect=lambda: torch.nn.Linear(2, 1, bias=False)):
            return predict.load_public_model("STNT", "base", checkpoint_root=self.root, **kwargs)

    def test_loads_verified_tensors_on_cpu_in_eval_mode(self):
        with patch.object(torch, "load", wraps=torch.load) as load:
            model, metadata = self.load()
        self.assertFalse(model.training)
        self.assertEqual(model.weight.device.type, "cpu")
        self.assertTrue(torch.equal(model.weight, torch.tensor([[1.0, 2.0]])))
        self.assertEqual(load.call_args.kwargs, {"weights_only": True, "map_location": "cpu"})
        self.assertEqual(metadata["selected_update"], 1200)
        self.assertEqual(metadata["file"], "STNT/base/seed_42.pth")
        self.assertNotIn(str(self.root), json.dumps(metadata))

    def test_hash_mismatch_stops_before_deserialization(self):
        self.entry["sha256"] = "f" * 64
        self.write_manifest()
        with patch.object(torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                self.load()
        load.assert_not_called()

    def test_size_mismatch_stops_before_deserialization(self):
        self.entry["bytes"] += 1
        self.write_manifest()
        with patch.object(torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "file size"):
                self.load()
        load.assert_not_called()

    def test_lfs_pointer_has_download_instruction(self):
        self.path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0" * 64 + "\nsize 1000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "separate Zenodo weight archive"):
            self.load()

    def test_missing_model_is_reported(self):
        with self.assertRaisesRegex(ValueError, "No public checkpoint"):
            predict.load_public_model("STNT", "real300", checkpoint_root=self.root)

    def test_missing_weight_file_is_reported(self):
        self.path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Public checkpoint is missing"):
            self.load()

    def test_missing_manifest_is_reported(self):
        (self.root / "manifest.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "manifest.json is missing"):
            self.load()

    def test_duplicate_identity_is_rejected(self):
        self.manifest["models"].append(dict(self.entry))
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate model identity"):
            self.load()

    def test_architecture_change_is_rejected(self):
        self.manifest["architecture"]["input_channels"] = 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "fast HoVerNet"):
            self.load()

    def test_paths_cannot_escape_checkpoint_root(self):
        for filename in ("../outside.pth", "/outside.pth", "C:/outside.pth", "STNT\\base\\seed_42.pth"):
            with self.subTest(filename=filename):
                self.entry["file"] = filename
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "relative path"):
                    self.load()

    def test_wrong_tensor_shape_is_rejected_by_strict_loading(self):
        self.save_state({"weight": torch.zeros(1, 3)})
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.load()

    def test_tensor_dtype_is_not_silently_cast(self):
        self.save_state({"weight": torch.zeros(1, 2, dtype=torch.float64)})
        with self.assertRaisesRegex(ValueError, "tensor dtypes"):
            self.load()

    def test_unexpected_state_key_is_rejected_by_strict_loading(self):
        self.save_state({"weight": torch.zeros(1, 2), "extra": torch.zeros(1)})
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            self.load()

    def test_nontensor_payload_is_rejected(self):
        self.save_state({"weight": torch.zeros(1, 2), "note": "not a tensor"})
        with self.assertRaisesRegex(ValueError, "tensor-only"):
            self.load()

    def test_tensor_statistics_must_match_manifest(self):
        for field in ("tensor_count", "tensor_bytes"):
            with self.subTest(field=field):
                self.entry[field] += 1
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "tensor count or byte size"):
                    self.load()
                self.entry[field] -= 1

    def test_unavailable_cuda_has_cpu_instruction(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "use --device cpu"):
                self.load(device="cuda")


class EmptySegmentationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_shapes = []

    def forward(self, image):
        self.batch_shapes.append(tuple(image.shape))
        shape = (len(image), 1, 164, 164)
        foreground = torch.full(shape, -10.0, device=image.device)
        return {"np": torch.cat([-foreground, foreground], dim=1),
                "hv": torch.zeros((len(image), 2, 164, 164), device=image.device)}


class PublicPredictCommandTests(unittest.TestCase):
    def assert_late_output_preserved(self, filename):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (8, 8), "white").save(image)
            out = root / "result"
            sentinel = b"created during inference"

            def late_output(*args, **kwargs):
                out.mkdir()
                (out / filename).write_bytes(sentinel)
                return object(), {}

            arguments = ["--image", str(image), "--tissue", "STNT", "--condition", "base",
                         "--output-dir", str(out)]
            with patch.object(predict, "load_public_model", side_effect=late_output), \
                    patch("src.eval.inference.predict_raw_map", return_value=(np.zeros((8, 8, 3), dtype=np.float32), 1)), \
                    patch("src.eval.inference.postprocess", return_value=(np.zeros((8, 8), dtype=np.int32), None)):
                with self.assertRaises(FileExistsError):
                    predict.main(arguments)
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, {filename: sentinel})

    def test_output_created_during_inference_is_preserved(self):
        self.assert_late_output_preserved("instances.npy")

    def test_later_output_collision_removes_only_new_reservations(self):
        for filename in ("raw_maps.npy", "prediction.json"):
            with self.subTest(filename=filename):
                self.assert_late_output_preserved(filename)

    def test_write_failure_removes_empty_reservations_and_keeps_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (8, 8), "white").save(image)
            out = root / "result"
            out.mkdir()
            (out / "existing.txt").write_bytes(b"keep")
            arguments = ["--image", str(image), "--tissue", "STNT", "--condition", "base",
                         "--output-dir", str(out)]
            with patch.object(predict, "load_public_model", return_value=(object(), {})), \
                    patch("src.eval.inference.predict_raw_map", return_value=(np.zeros((8, 8, 3), dtype=np.float32), 1)), \
                    patch("src.eval.inference.postprocess", return_value=(np.zeros((8, 8), dtype=np.int32), None)), \
                    patch.object(np, "save", side_effect=OSError("Write failed")):
                with self.assertRaisesRegex(OSError, "Write failed"):
                    predict.main(arguments)
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, {"existing.txt": b"keep"})

    def test_help_does_not_load_a_model(self):
        with patch.object(predict, "load_public_model") as load, contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as stopped:
                predict.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--checkpoint-dir", output.getvalue())
        load.assert_not_called()

    def test_external_rgb_conversion_tiling_postprocessing_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "external_grayscale.png"
            Image.fromarray(np.zeros((273, 175), dtype=np.uint8)).save(image)
            out = root / "result"
            model = EmptySegmentationModel()
            metadata = {"tissue": "STNT", "condition": "base", "seed": 42, "sha256": "0" * 64}
            arguments = ["--image", str(image), "--tissue", "STNT", "--condition", "base",
                         "--output-dir", str(out), "--batch-size", "3"]
            with patch.object(predict, "load_public_model", return_value=(model, metadata)), contextlib.redirect_stdout(io.StringIO()):
                predict.main(arguments)
            instances = np.load(out / "instances.npy", allow_pickle=False)
            raw = np.load(out / "raw_maps.npy", allow_pickle=False)
            report = json.loads((out / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(instances.shape, (273, 175))
            self.assertTrue(np.issubdtype(instances.dtype, np.integer))
            self.assertFalse(instances.any())
            self.assertEqual(raw.shape, (273, 175, 3))
            self.assertEqual(raw.dtype, np.float32)
            self.assertEqual(model.batch_shapes, [(3, 3, 256, 256), (1, 3, 256, 256)])
            self.assertEqual((report["patch_count"], report["batch_count"], report["instance_count"]), (4, 2, 0))
            self.assertEqual(report["raw_map_channels"], ["NP", "H", "V"])
            self.assertNotIn(str(root), json.dumps(report))
            snapshot = {p.name: p.read_bytes() for p in out.iterdir()}
            with patch.object(predict, "load_public_model") as load:
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    predict.main(arguments)
            load.assert_not_called()
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in out.iterdir()})
            with patch.object(predict, "load_public_model", return_value=(model, metadata)), contextlib.redirect_stdout(io.StringIO()):
                predict.main(arguments + ["--overwrite"])
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
