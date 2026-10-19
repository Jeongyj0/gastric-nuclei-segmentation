"""Public evaluation wiring tests; no claims about pretrained model quality."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from src import study
from src.eval import public_evaluate as public
from src.eval import public_statistics
from src.eval.metrics import score_raw_prediction, summarize
from src.eval.statistics import paired_patient_bootstrap
from src.eval.predict import ARCHITECTURE


class PublicEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        root_patch = patch.object(study, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir()
        self.entries = [dict(tissue=t, condition=c, seed=42, selected_update=1200,
                            file=f"{t}/{c}/seed_42.pth", sha256="1" * 64, bytes=1,
                            tensor_count=1, tensor_bytes=4, source_checkpoint_sha256="2" * 64)
                        for t in study.TISSUES for c in study.CONDITIONS]
        self.manifest = self.checkpoints / "manifest.json"
        self.manifest.write_text(json.dumps({"format_version": 1, "architecture": ARCHITECTURE,
                                             "models": self.entries}), encoding="utf-8")
        self.data = self.root / "data/manifests"
        self.rows_by_tissue = {}
        for tissue in study.TISSUES:
            rows = [dict(id=f"image_{i:03d}", pid=f"patient_{i // 3:03d}", split="test",
                         source="real", cancer_type=tissue, image_path="image.png", inst_mask_path="mask.npy")
                    for i in range(75)]
            self.rows_by_tissue[tissue] = rows
            folder = self.data / tissue
            folder.mkdir(parents=True)
            self.write_csv(folder / "test.csv", rows)
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(self.root / "image.png")
        self.truth = np.zeros((32, 32), dtype=np.int32)
        self.truth[8:24, 8:24] = 7
        np.save(self.root / "mask.npy", self.truth)
        self.raw = np.zeros((32, 32, 3), dtype=np.float32)
        self.raw[..., 0] = (self.truth > 0).astype(np.float32) * .9
        self.config = self.root / "analysis.json"
        self.config.write_text(json.dumps({"training_seeds": [42], "bootstrap": {
            "repetitions": 30, "seed": 42, "confidence": .95,
            "method": "paired_patient_cluster_percentile_image_weighted",
            "multiplicity": "exploratory_pointwise_intervals_no_significance_claims"}}), encoding="utf-8")

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def invoke(self, *args):
        metadata = {**self.entries[0], "format_version": 1, "architecture": ARCHITECTURE}
        with patch.object(public, "load_public_model", return_value=(object(), metadata)) as loader, \
                patch.object(public, "evaluation_sources", return_value={"fixture.py": "a" * 64}), \
                patch("src.eval.inference.predict_raw_map", return_value=(self.raw, 1)), \
                patch("src.eval.inference.postprocess", return_value=(self.truth.copy(), None)):
            public.main(["--tissue", "STNT", "--condition", "base", "--manifest-root", "data/manifests",
                         "--checkpoint-dir", str(self.checkpoints), "--output-dir", "outputs/eval", *args])
            return loader

    def test_cli_metrics_froc_and_statistics_loader_without_training_records(self):
        loader = self.invoke("--froc")
        self.assertEqual(loader.call_count, 1)
        folder = self.root / "outputs/eval/STNT/base/seed_42"
        summary = json.loads((folder / "test_summary.json").read_text())
        reference = score_raw_prediction(self.truth, self.truth, self.raw[..., 0])
        self.assertEqual(summary["images"], 75)
        self.assertAlmostEqual(summary["mean_pq"], reference["pq"])
        self.assertEqual(summary["froc"]["TP"], 75)
        self.assertEqual(summary["froc"]["pq_audit_pass_count"], 75)
        self.assertTrue((folder / "data/froc_curves/STNT_base.csv").is_file())
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "pretrained").exists())
        self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "completed")
        # The original research path still rejects public-state evaluation records.
        from src.eval.statistics import load_evaluation
        with self.assertRaisesRegex(ValueError, "checkpoint kind"):
            load_evaluation(Path("outputs/eval"), "STNT", "base", 42, self.rows_by_tissue["STNT"],
                            study.sha256(self.data / "STNT/test.csv"))

    def test_existing_output_is_preserved(self):
        self.invoke()
        path = self.root / "outputs/eval/STNT/base/seed_42/test_summary.json"
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.invoke()
        self.assertEqual(path.read_bytes(), before)

    def test_froc_without_any_predictions_is_valid_json(self):
        self.truth[:] = 0
        self.raw[:] = 0
        np.save(self.root / "mask.npy", self.truth)
        self.invoke("--froc")
        path = self.root / "outputs/eval/STNT/base/seed_42/test_summary.json"
        summary = json.loads(path.read_text())
        self.assertEqual(summary["froc"]["score"], "inf")
        self.assertEqual(summary["froc"]["TP"], 0)
        self.assertEqual(summary["froc"]["FP"], 0)

    def test_nonfinite_inference_records_failure(self):
        self.raw[0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            self.invoke()
        folder = self.root / "outputs/eval/STNT/base/seed_42"
        self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "failed")
        self.assertFalse((folder / "test_summary.json").exists())

    def write_statistics_fixtures(self):
        for entry in self.entries:
            tissue, condition = entry["tissue"], entry["condition"]
            rows = self.rows_by_tissue[tissue]
            folder = self.root / "outputs/stats_input" / tissue / condition / "seed_42"
            (folder / "private").mkdir(parents=True)
            metrics = []
            for i, row in enumerate(rows):
                result = score_raw_prediction(self.truth, self.truth if i % 2 else np.zeros_like(self.truth), self.raw[..., 0])
                result.update(id=row["id"], pid=row["pid"], patches=1)
                metrics.append(result)
            self.write_csv(folder / "private/test_metrics.csv", metrics)
            inputs = [{**{k: r[k] for k in ("id", "pid", "image_path", "inst_mask_path")},
                       "image_sha256": "3" * 64, "mask_sha256": "4" * 64} for r in rows]
            study.write_json(folder / "private/inputs.json", inputs)
            summary = summarize(metrics)
            summary.update(tissue=tissue, condition=condition, seed=42, checkpoint=public.CHECKPOINT_KIND,
                           public_model=entry, checkpoint_sha256=entry["sha256"], selected_update=entry["selected_update"],
                           checkpoint_manifest_sha256=study.sha256(self.manifest),
                           test_manifest_sha256=study.sha256(self.data / tissue / "test.csv"),
                           metrics_sha256=study.sha256(folder / "private/test_metrics.csv"),
                           input_records_sha256=study.sha256(folder / "private/inputs.json"),
                           evaluation_sources={"fixture.py": "a" * 64})
            study.write_json(folder / "test_summary.json", summary)
            study.write_json(folder / "status.json", {"status": "completed"})

    def statistics(self):
        return public_statistics.summarize_public("outputs/stats_input", "data/manifests",
                                                  self.checkpoints, "analysis.json", "outputs/stats")

    def test_statistics_pairs_inputs_and_uses_existing_bootstrap(self):
        self.write_statistics_fixtures()
        report = self.statistics()
        reference = paired_patient_bootstrap(np.zeros((1, 75)),
                                              [r["pid"] for r in self.rows_by_tissue["STNT"]], 30, 42)
        self.assertEqual(len(report["contrasts"]), 16)
        for result in report["contrasts"]:
            for key, value in reference.items():
                self.assertEqual(result[key], value)
        self.assertTrue((self.root / "outputs/stats/statistics.json").is_file())

    def test_statistics_rejects_modified_metrics(self):
        self.write_statistics_fixtures()
        path = self.root / "outputs/stats_input/STNT/base/seed_42/private/test_metrics.csv"
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.statistics()

    def test_statistics_rejects_model_source_and_input_changes(self):
        self.write_statistics_fixtures()
        path = self.root / "outputs/stats_input/STNT/add/seed_42/test_summary.json"
        original = json.loads(path.read_text())
        changes = (("checkpoint_sha256", "9" * 64, "release manifest"),
                   ("evaluation_sources", {"different.py": "a" * 64}, "source versions"),
                   ("input_records_sha256", "0" * 64, "input records"))
        for key, value, error in changes:
            with self.subTest(key=key):
                study.write_json(path, {**original, key: value})
                with self.assertRaisesRegex(ValueError, error):
                    self.statistics()
        study.write_json(path, original)
        inputs_path = path.parent / "private/inputs.json"
        inputs = json.loads(inputs_path.read_text())
        inputs[0]["mask_sha256"] = "0" * 64
        study.write_json(inputs_path, inputs)
        study.write_json(path, {**original, "input_records_sha256": study.sha256(inputs_path)})
        with self.assertRaisesRegex(ValueError, "different image or mask"):
            self.statistics()


if __name__ == "__main__":
    unittest.main()
