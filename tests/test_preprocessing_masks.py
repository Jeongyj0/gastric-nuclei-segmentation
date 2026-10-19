from pathlib import Path
import contextlib
import csv
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.preprocessing import make_masks


def write_labels(path, objects):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": {"file": {"object": objects}}}), encoding="utf-8-sig")


class PolygonMaskTests(unittest.TestCase):
    def test_coordinate_rounding_and_label_fallback(self):
        obj = {"label_nm": "Normal", "coordinate": [{"x": 0.5, "y": 1.5}, [2.5, 3.5], [4.5, 5.5]]}
        self.assertEqual(make_masks.get_label(obj), "Normal")
        self.assertEqual(make_masks.get_polygon(obj), [(0, 2), (2, 4), (4, 6)])

    def test_first_object_wins_clipping_empty_skip_and_single_pixel_nuclei(self):
        first = {"label": "Stroma", "coordinate": [[0, 0], [3, 0], [3, 3], [0, 3]]}
        objects = [first, {"label": "Immune", "coordinate": [[2, 2], [5, 2], [5, 5], [2, 5]]},
                   first, {"label": "Normal", "coordinate": [[-5, -5], [-2, -5], [-2, -2]]},
                   {"label": "Tumor", "coordinate": [[7, 7], [7, 7], [7, 7]]},
                   {"label": "Normal", "coordinate": [[-2, 6], [1, 6], [1, 9], [-2, 9]]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            write_labels(path, objects)
            instances, classes = make_masks.json_to_masks(path, 8, 8)
        expected = np.zeros((8, 8), dtype=np.int32)
        expected[:4, :4] = 1
        expected[2:4, 4:6] = 2
        expected[4:6, 2:6] = 2
        expected[7, 7] = 3
        expected[6:, :2] = 4
        expected_classes = np.zeros((8, 8), dtype=np.uint8)
        for instance, category in ((1, 1), (2, 2), (3, 4), (4, 3)):
            expected_classes[expected == instance] = category
        np.testing.assert_array_equal(instances, expected)
        np.testing.assert_array_equal(classes, expected_classes)
        self.assertEqual(instances.dtype, np.int32)
        self.assertEqual(classes.dtype, np.uint8)

    def test_empty_annotation_produces_zero_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            write_labels(path, [])
            instances, classes = make_masks.json_to_masks(path, 3, 5)
        self.assertEqual(instances.shape, (3, 5))
        self.assertFalse(instances.any())
        self.assertFalse(classes.any())

    def test_unknown_classes_and_invalid_polygons_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            for obj, message in (({"label": "Other", "coordinate": [[1, 1], [2, 2], [3, 3]]}, "Unknown label"),
                                 ({"label": "Tumor", "coordinate": [[1, 1], [2, 2]]}, "Invalid polygon")):
                with self.subTest(message=message):
                    write_labels(path, [obj])
                    with self.assertRaisesRegex(ValueError, message):
                        make_masks.json_to_masks(path, 8, 8)


class MaskCommandTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = {
            "id": "0001", "source": "real", "split": "train", "cancer_type": "STNT",
            "image_path": "data/images/image.png", "label_path": "data/labels/image.json",
            "inst_mask_path": "data/masks/instances.npy", "class_mask_path": "data/masks/classes.npy",
            "extra": "0007",
        }
        image = self.root / self.row["image_path"]
        image.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(image)
        write_labels(self.root / self.row["label_path"], [
            {"label": "Tumor", "coordinate": [[2, 2], [4, 2], [4, 4], [2, 4]]}])
        self.meta = self.root / "data/selected_data.csv"
        self.write_metadata()

    def write_metadata(self, rows=None):
        with self.meta.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.row))
            writer.writeheader()
            writer.writerows([self.row] if rows is None else rows)

    def run_command(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            make_masks.main(["--root", str(self.root), *extra])
        return json.loads(output.getvalue())

    def output_snapshot(self):
        paths = (self.root / self.row["inst_mask_path"], self.root / self.row["class_mask_path"], self.root / "data/meta.csv")
        return {str(path.relative_to(self.root)): path.read_bytes() for path in paths if path.is_file()}

    def test_default_paths_preserve_metadata_and_mask_types(self):
        report = self.run_command()
        instances = np.load(self.root / self.row["inst_mask_path"], allow_pickle=False)
        classes = np.load(self.root / self.row["class_mask_path"], allow_pickle=False)
        expected = np.zeros((8, 8), dtype=np.int32)
        expected[2:5, 2:5] = 1
        np.testing.assert_array_equal(instances, expected)
        np.testing.assert_array_equal(classes, (expected * 4).astype(np.uint8))
        self.assertEqual(instances.dtype, np.int32)
        self.assertEqual(classes.dtype, np.uint8)
        with (self.root / "data/meta.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, list(self.row))
            self.assertEqual(list(reader), [self.row])
        self.assertEqual(report, {"processed": 1, "created": 1, "reused": 0, "output_meta": "data/meta.csv"})

    def test_existing_masks_require_explicit_validated_reuse(self):
        self.run_command()
        before = self.output_snapshot()
        with self.assertRaisesRegex(FileExistsError, "reuse-existing"):
            self.run_command()
        report = self.run_command("--reuse-existing")
        self.assertEqual((report["created"], report["reused"]), (0, 1))
        self.assertEqual(self.output_snapshot(), before)

    def test_changed_values_or_dtype_are_never_reused_or_replaced(self):
        self.run_command()
        path = self.root / self.row["inst_mask_path"]
        original = np.load(path, allow_pickle=False)
        for changed in (original + 1, original.astype(np.int64)):
            with self.subTest(dtype=str(changed.dtype), changed=bool(changed[0, 0])):
                np.save(path, changed, allow_pickle=False)
                before = self.output_snapshot()
                with self.assertRaisesRegex(ValueError, "Existing masks differ"):
                    self.run_command("--reuse-existing")
                self.assertEqual(self.output_snapshot(), before)

    def test_partial_mask_pair_is_rejected(self):
        path = self.root / self.row["inst_mask_path"]
        path.parent.mkdir(parents=True)
        np.save(path, np.zeros((8, 8), dtype=np.int32), allow_pickle=False)
        before = self.output_snapshot()
        with self.assertRaisesRegex(FileExistsError, "complete pair"):
            self.run_command("--reuse-existing")
        self.assertEqual(self.output_snapshot(), before)

    def test_paths_must_be_relative_and_stay_inside_root(self):
        for value in ("../outside.npy", str(self.root / "inside.npy"), "C:/outside.npy", "."):
            with self.subTest(value=value):
                self.row["inst_mask_path"] = value
                self.write_metadata()
                with self.assertRaises(ValueError):
                    self.run_command()
        self.assertFalse((self.root / "data/masks").exists())
        self.assertFalse((self.root / "data/meta.csv").exists())

    def test_metadata_cannot_replace_the_selection_input(self):
        before = self.meta.read_bytes()
        with self.assertRaisesRegex(ValueError, "must not replace input"):
            self.run_command("--output-meta", "data/selected_data.csv")
        self.assertEqual(self.meta.read_bytes(), before)
        self.assertFalse((self.root / "data/masks").exists())

    def test_duplicate_ids_and_mask_destinations_fail_before_writing(self):
        self.write_metadata([self.row, dict(self.row)])
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            self.run_command()
        self.row["class_mask_path"] = self.row["inst_mask_path"]
        self.write_metadata()
        with self.assertRaisesRegex(ValueError, "distinct path"):
            self.run_command()
        self.assertFalse((self.root / "data/masks").exists())

    def test_invalid_selection_values_fail_before_writing(self):
        original = dict(self.row)
        for key in ("source", "split", "cancer_type"):
            with self.subTest(key=key):
                self.row = dict(original)
                self.row[key] = "unknown"
                self.write_metadata()
                with self.assertRaisesRegex(ValueError, f"unsupported {key}"):
                    self.run_command()
        self.assertFalse((self.root / "data/masks").exists())
        self.assertFalse((self.root / "data/meta.csv").exists())

    def test_conflicting_output_metadata_is_preserved(self):
        other = dict(self.row, id="different")
        output_meta = self.root / "data/meta.csv"
        with output_meta.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(other))
            writer.writeheader()
            writer.writerow(other)
        before = output_meta.read_bytes()
        with self.assertRaisesRegex(ValueError, "output metadata differs"):
            self.run_command("--reuse-existing")
        self.assertEqual(output_meta.read_bytes(), before)
        self.assertFalse((self.root / "data/masks").exists())

    def test_mask_reservation_collision_preserves_existing_file(self):
        first, second = self.root / "first.npy", self.root / "second.npy"
        second.write_bytes(b"existing mask")
        with self.assertRaises(FileExistsError):
            make_masks._save_pair((first, second), (np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.uint8)))
        self.assertFalse(first.exists())
        self.assertEqual(second.read_bytes(), b"existing mask")

    def test_metadata_write_failure_removes_partial_file_and_allows_retry(self):
        with patch.object(csv.DictWriter, "writerows", side_effect=OSError("Write failed")):
            with self.assertRaisesRegex(OSError, "Write failed"):
                self.run_command()
        self.assertFalse((self.root / "data/meta.csv").exists())
        report = self.run_command("--reuse-existing")
        self.assertEqual(report["reused"], 1)
        self.assertTrue((self.root / "data/meta.csv").is_file())


if __name__ == "__main__":
    unittest.main()
