from pathlib import Path
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import make_patient_disjoint_split as patient
from src.data import make_splits
from src import study


def metadata_fixture():
    rows = []
    for tissue in patient.CANCERS:
        for split, count in patient.TARGET.items():
            for index in range(count):
                rows.append({"id": f"{tissue}_{split}_{index:04d}", "pid": f"shared_{split}",
                             "source": "real", "split": split, "cancer_type": tissue,
                             "image_path": "data/image.png", "label_path": "data/label.json",
                             "inst_mask_path": "data/instance.npy", "class_mask_path": "data/class.npy"})
        for index in range(600):
            rows.append({"id": f"{tissue}_synthetic_{index:04d}", "pid": pd.NA,
                         "source": "synthetic", "split": "train", "cancer_type": tissue,
                         "image_path": "data/image.png", "label_path": "data/label.json",
                         "inst_mask_path": "data/instance.npy", "class_mask_path": "data/class.npy"})
    return pd.DataFrame(rows)


def write_inputs(root, meta, relative="data/meta_patient_disjoint.csv"):
    (root / "data").mkdir(exist_ok=True)
    for name in ("image.png", "instance.npy", "class.npy"):
        (root / "data" / name).write_bytes(b"fixture")
    (root / "data/label.json").write_text('{"content":{"clinical":{"pid":"fixture_patient"}}}', encoding="utf-8")
    meta.to_csv(root / relative, index=False)


class StudyManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = metadata_fixture()

    def test_counts_shared_sets_and_real300_row_order(self):
        manifests = make_splits.build_manifests(self.meta)
        self.assertEqual(len(manifests), 28)
        self.assertFalse(any("replace25" in name or "replace75" in name for name in manifests))
        for tissue in patient.CANCERS:
            frames = {name: manifests[f"{tissue}/{name}/train.csv"] for name in make_splits.CONDITIONS}
            for name, expected in make_splits.CONDITIONS.items():
                counts = frames[name]["source"].value_counts()
                self.assertEqual((counts.get("real", 0), counts.get("synthetic", 0)), expected)
            self.assertEqual(set(frames["base"]["id"]), set(frames["add"].loc[frames["add"]["source"] == "real", "id"]))
            self.assertEqual(set(frames["replace100"]["id"]), set(frames["add"].loc[frames["add"]["source"] == "synthetic", "id"]))
            self.assertTrue(set(frames["replace50"]["id"]) <= set(frames["add"]["id"]))
            expected = frames["replace50"].loc[frames["replace50"]["source"] == "real"].reset_index(drop=True)
            pd.testing.assert_frame_equal(frames["real300"].reset_index(drop=True), expected)
            for split in ("val", "test"):
                frame = manifests[f"{tissue}/{split}.csv"]
                self.assertEqual(len(frame), 75)
                self.assertEqual(frame["id"].tolist(), sorted(frame["id"]))
                self.assertTrue(frame["source"].eq("real").all())

    def test_input_row_order_does_not_change_selected_ids_or_manifest_order(self):
        original = make_splits.build_manifests(self.meta)
        reordered = make_splits.build_manifests(self.meta.sample(frac=1, random_state=7))
        for name in original:
            pd.testing.assert_frame_equal(original[name].reset_index(drop=True), reordered[name].reset_index(drop=True))

    def test_cross_tissue_patient_leak_is_rejected(self):
        meta = self.meta.copy()
        row = meta.index[(meta["cancer_type"] == "STIN") & (meta["split"] == "val")][0]
        meta.loc[row, "pid"] = "shared_train"
        with self.assertRaisesRegex(ValueError, "Patient split overlap"):
            make_splits.build_manifests(meta)

    def test_missing_patient_ids_and_wrong_counts_are_rejected(self):
        for value in (pd.NA, np.nan, "", "nan"):
            with self.subTest(value=str(value)):
                meta = self.meta.copy()
                meta.loc[0, "pid"] = value
                with self.assertRaisesRegex(ValueError, "PID"):
                    make_splits.validate_meta(meta)
        with self.assertRaisesRegex(ValueError, "expected 600"):
            make_splits.validate_meta(self.meta.iloc[1:].copy())
        with self.assertRaisesRegex(ValueError, "required"):
            make_splits.validate_meta(self.meta.drop(columns="inst_mask_path"))

    def test_cli_writes_five_conditions_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, self.meta)
            arguments = ["--root", str(root)]
            with contextlib.redirect_stdout(io.StringIO()):
                make_splits.main(arguments)
            output = root / "data/manifests"
            snapshot = {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()}
            self.assertEqual(len(snapshot), 28)
            for tissue in patient.CANCERS:
                full = pd.read_csv(output / tissue / "replace50/train.csv", keep_default_na=False)
                small = pd.read_csv(output / tissue / "real300/train.csv", keep_default_na=False)
                pd.testing.assert_frame_equal(small, full.loc[full["source"] == "real"].reset_index(drop=True))
            with patch.object(study, "ROOT", root):
                audit = study.audit_manifests({"manifest_root": "data/manifests"})
                self.assertEqual(len(audit["manifest_sha256"]), 28)
                self.assertTrue(all(count == 0 for count in audit["patient_overlap"].values()))
            with self.assertRaises(FileExistsError):
                make_splits.main(arguments)
            self.assertEqual(snapshot, {path.relative_to(output).as_posix(): path.read_bytes() for path in output.rglob("*") if path.is_file()})

    def test_partial_manifest_write_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, self.meta)
            original = pd.DataFrame.to_csv
            calls = []
            def fail_after_first(frame, destination, *args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    destination.write("partial")
                    raise OSError("Simulated write failure")
                return original(frame, destination, *args, **kwargs)
            with patch.object(pd.DataFrame, "to_csv", fail_after_first):
                with self.assertRaisesRegex(OSError, "Simulated"):
                    make_splits.main(["--root", str(root)])
            self.assertFalse((root / "data/manifests").exists())
            self.assertTrue((root / "data/meta_patient_disjoint.csv").is_file())

    def test_cli_rejects_external_paths_and_missing_input_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_inputs(root, self.meta)
            for value in ("../outside", str(root.parent / "outside"), "C:/outside"):
                with self.subTest(path=value), self.assertRaises(ValueError):
                    make_splits.main(["--root", str(root), "--output-dir", value])
            invalid = self.meta.copy()
            invalid.loc[0, "image_path"] = "../outside.png"
            invalid.to_csv(root / "data/meta_patient_disjoint.csv", index=False)
            with self.assertRaises(ValueError):
                make_splits.main(["--root", str(root)])
            self.meta.to_csv(root / "data/meta_patient_disjoint.csv", index=False)
            (root / "data/instance.npy").unlink()
            with self.assertRaises(FileNotFoundError):
                make_splits.main(["--root", str(root)])
            self.assertFalse((root / "data/manifests").exists())

    def test_metadata_preserves_zero_prefixed_ids_without_changing_other_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = self.meta.copy()
            meta.loc[0, "id"] = "0001"
            meta.loc[0, "pid"] = "0007"
            meta["measurement"] = 10
            write_inputs(root, meta)
            actual = patient.read_meta(root / "data/meta_patient_disjoint.csv", root, require_pid=True)
            self.assertEqual(actual.loc[0, "id"], "0001")
            self.assertEqual(actual.loc[0, "pid"], "0007")
            self.assertTrue(pd.api.types.is_integer_dtype(actual["measurement"]))
            self.assertTrue(actual.loc[actual["source"] == "synthetic", "pid"].isna().all())


class PatientAllocationTests(unittest.TestCase):
    def test_infeasible_patient_counts_fail_without_splitting_patients(self):
        real = metadata_fixture().query("source == 'real'").copy()
        real["pid"] = "one_patient"
        with self.assertRaisesRegex(RuntimeError, "infeasible"):
            patient.assign_patients(real)

    def test_solver_allocates_shared_patients_globally_and_preserves_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labels").mkdir()
            for index in range(10):
                (root / f"labels/{index}.json").write_text(json.dumps({"content": {"clinical": {"pid": f"shared_{index}"}}}), encoding="utf-8")
            meta = metadata_fixture()
            for tissue in patient.CANCERS:
                rows = meta.index[(meta["cancer_type"] == tissue) & (meta["source"] == "real")]
                meta.loc[rows, "label_path"] = [f"labels/{index // 75}.json" for index in range(750)]
            result = patient.make_patient_disjoint_meta(meta, root)
            self.assertEqual(result["id"].tolist(), meta["id"].tolist())
            self.assertEqual(result["original_split"].tolist(), meta["split"].tolist())
            real = result[result["source"] == "real"]
            self.assertTrue(real.groupby("pid")["split"].nunique().eq(1).all())
            for tissue in patient.CANCERS:
                self.assertEqual(real[real["cancer_type"] == tissue]["split"].value_counts().to_dict(), patient.TARGET)
            synthetic = result[result["source"] == "synthetic"]
            self.assertTrue(synthetic["split"].eq("train").all())
            self.assertTrue(synthetic["pid"].isna().all())

    def test_invalid_json_patient_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            for pid in (None, "", "NaN", [], float("nan")):
                with self.subTest(pid=str(pid)):
                    path.write_text(json.dumps({"content": {"clinical": {"pid": pid}}}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "valid.*pid"):
                        patient.load_pid(path)

    def test_json_numeric_and_string_patient_ids_share_a_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            for value in (7, 7.0, "7", " 7 "):
                path.write_text(json.dumps({"content": {"clinical": {"pid": value}}}), encoding="utf-8")
                self.assertEqual(patient.load_pid(path), "7")
            path.write_text(json.dumps({"content": {"clinical": {"pid": "0007"}}}), encoding="utf-8")
            self.assertEqual(patient.load_pid(path), "0007")

    def test_patient_output_is_exclusive_and_partial_write_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = metadata_fixture()
            write_inputs(root, meta, "data/meta.csv")
            output = root / "data/meta_patient_disjoint.csv"
            arguments = ["--root", str(root)]
            output.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                patient.main(arguments)
            self.assertEqual(output.read_bytes(), b"keep")
            output.unlink()
            def fail_write(frame, stream, *args, **kwargs):
                stream.write("partial")
                raise OSError("Simulated write failure")
            with patch.object(patient, "make_patient_disjoint_meta", return_value=meta), patch.object(pd.DataFrame, "to_csv", fail_write):
                with self.assertRaisesRegex(OSError, "Simulated"):
                    patient.main(arguments)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
