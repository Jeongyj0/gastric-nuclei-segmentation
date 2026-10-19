import copy
import ast
from types import SimpleNamespace
import unittest
import numpy as np
from src.study import ROOT, training_settings, project_path, POLICY
from config_fixture import read_study_config
from src.eval.metrics import score_instances, score_raw_prediction, summarize
from src.eval.statistics import paired_patient_bootstrap
from src.train.control import validation_due, learning_rate_after, select_checkpoints


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.zeros((12, 16), dtype=np.int32)
        self.truth[1:5, 1:5] = 7
        self.truth[7:10, 10:15] = 22
        self.empty = np.zeros_like(self.truth)

    def test_perfect_noncontiguous_ids(self):
        score = score_instances(self.truth, self.truth * 9)
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (2, 0, 0))
        self.assertAlmostEqual(score["pq"], 1, places=5)
        self.assertEqual(score["aji"], 1)

    def test_empty_prediction_is_included(self):
        score = score_instances(self.truth, self.empty)
        self.assertEqual((score["pq"], score["aji"], score["recall"]), (0, 0, 0))
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (0, 0, 2))

    def test_empty_truth_and_both_empty(self):
        for truth, pred, fp in ((self.empty, self.truth, 2), (self.empty, self.empty, 0)):
            result = score_instances(truth, pred)
            self.assertTrue(all(np.isfinite(v) for v in result.values()))
            self.assertEqual((result["pq"], result["aji_plus"], result["fp"]), (0, 0, fp))

    def test_full_foreground_without_background(self):
        truth = np.ones((5, 5), dtype=np.int32) * 38
        score = score_instances(truth, truth)
        self.assertEqual(score["tp"], 1)
        self.assertAlmostEqual(score["pq"], 1, places=5)

    def test_iou_exactly_half_is_unmatched(self):
        truth = np.zeros((4, 6), dtype=np.int32)
        truth[1:3, 1:5] = 1
        pred = truth.copy()
        pred[:, 3:] = 0
        result = score_instances(truth, pred)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (0, 1, 1))

    def test_np_threshold_and_image_weighting(self):
        one = np.array([[1, 0]], dtype=np.int32)
        first = score_raw_prediction(one, one, np.array([[0.5, 0.49]]))
        second = score_raw_prediction(self.truth, self.empty, np.zeros_like(self.truth))
        summary = summarize([first, second])
        self.assertEqual(first["np_dice"], 1)
        self.assertAlmostEqual(summary["mean_pq"], first["pq"] / 2)
        self.assertNotAlmostEqual(summary["mean_pq"], summary["pooled_pq"])
        self.assertEqual(summary["empty_pred"], 1)

    def test_invalid_values_fail(self):
        for array in (self.truth - 1, self.truth.astype(float)):
            with self.assertRaises(ValueError):
                score_instances(array, self.empty)
        with self.assertRaises(ValueError):
            score_raw_prediction(self.truth, self.truth, np.full(self.truth.shape, np.nan))


class BootstrapTests(unittest.TestCase):
    def test_image_weighted_estimate_not_patient_mean(self):
        result = paired_patient_bootstrap([[0, 0, 1], [0, 0, 1]], ["a", "a", "b"], 2000, 4)
        self.assertAlmostEqual(result["mean_difference"], 1 / 3)
        self.assertEqual(result["training_seed_sd"], 0)
        self.assertEqual((result["images"], result["patients"]), (3, 2))
        self.assertEqual((result["ci_low"], result["ci_high"]), (0, 1))

    def test_seed_uncertainty_kept_separate(self):
        result = paired_patient_bootstrap([[0.2, 0.2], [-0.2, -0.2]], ["a", "b"], 50, 7)
        self.assertEqual((result["ci_low"], result["ci_high"]), (0, 0))
        self.assertGreater(result["training_seed_sd"], 0)

    def test_reproducibility_and_single_seed(self):
        args = ([[0, 0.2, -0.3, 1]], ["a", "a", "b", "c"], 120, 43)
        self.assertEqual(paired_patient_bootstrap(*args), paired_patient_bootstrap(*args))
        self.assertIsNone(paired_patient_bootstrap(*args)["training_seed_sd"])

    def test_nan_or_missing_alignment_rejected(self):
        for differences, patients in (([[float("nan")]], ["a"]), ([[1, 2]], ["a"])):
            with self.assertRaises(ValueError):
                paired_patient_bootstrap(differences, patients, 10, 1)


class ControlTests(unittest.TestCase):
    def test_validation_at_budget_end_without_extra_step(self):
        self.assertEqual([u for u in range(1, 12) if validation_due(u, 11, 4)], [4, 8, 11])

    def test_schedule_based_on_completed_optimizer_updates(self):
        schedule = {"initial_lr": 1e-4, "gamma": 0.1, "step_size_updates": 3}
        for completed, expected in ((0, 1e-4), (2, 1e-4), (3, 1e-5), (6, 1e-6)):
            self.assertAlmostEqual(learning_rate_after(completed, schedule), expected)

    def test_checkpoint_ties_and_separate_metrics(self):
        best = {"pq": None, "np_dice": None}
        self.assertEqual(select_checkpoints(best, 0, 0.4, 3), ["best_pq.pth", "best_np_dice.pth"])
        self.assertEqual(select_checkpoints(best, 0, 0.4, 6), [])
        self.assertEqual(select_checkpoints(best, 0.1, 0.3, 9), ["best_pq.pth"])
        self.assertEqual(best["np_dice"]["update"], 3)

    def test_config_values_required_without_invented_defaults(self):
        config = read_study_config()
        config["training_seeds"] = None
        with self.assertRaises(ValueError):
            training_settings(config)

    def test_config_validation(self):
        config = read_study_config()
        config.update(training_seeds=[101, 202], batch_size=1, max_optimizer_updates=11,
                      validation_interval_updates=4,
                      lr_schedule={"kind": "step", "initial_lr": 1e-4, "step_size_updates": 6, "gamma": 0.1},
                      encoder_freeze_schedule={"freeze_updates": 0})
        config.pop("early_stopping", None)
        config["policy"] = copy.deepcopy(POLICY)
        self.assertEqual(training_settings(config)["max_optimizer_updates"], 11)
        for key, invalid in (("training_seeds", [1, 1]), ("batch_size", 0),
                             ("validation_interval_updates", 12)):
            bad = copy.deepcopy(config)
            bad[key] = invalid
            with self.assertRaises(ValueError):
                training_settings(bad)

    def test_external_paths_rejected(self):
        with self.assertRaises(ValueError):
            project_path("../research/data")


class KernelMathTests(unittest.TestCase):
    def test_source_kernel_xy_ramps_without_claiming_torch_execution(self):
        from scipy.signal import correlate2d
        source = ROOT / "source_core/hovernet/models/hovernet/utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        loss = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "msge_loss")
        kernel = next(node for node in loss.body if isinstance(node, ast.FunctionDef)
                      and node.name == "get_sobel_kernel")
        adapter = SimpleNamespace(arange=lambda start, stop, dtype, device, requires_grad:
                                  np.arange(start, stop, dtype=dtype),
                                  meshgrid=lambda a, b, indexing: np.meshgrid(a, b, indexing=indexing))
        namespace = {"torch": adapter, "pred": SimpleNamespace(dtype=np.float32, device="cpu")}
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), "exec"), namespace)
        kernel_x, kernel_y = namespace["get_sobel_kernel"](5)
        y, x = np.mgrid[:19, :21].astype(np.float32)
        for kernel, ramp, expected in ((kernel_x, x, 12), (kernel_x, y, 0),
                                       (kernel_y, x, 0), (kernel_y, y, 12)):
            np.testing.assert_allclose(correlate2d(ramp.astype(np.float64),
                                                   kernel.astype(np.float64), mode="valid"),
                                       expected, atol=1e-6)


class PostprocessTests(unittest.TestCase):
    def test_empty_map_and_warning_function_preserved(self):
        import importlib.util
        import warnings
        before = warnings.warn
        spec = importlib.util.spec_from_file_location(
            "study_postprocess", ROOT / "source_core/hovernet/models/hovernet/post_proc.py")
        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            spec.loader.exec_module(module)
            labels, info = module.process(np.zeros((64, 64, 3), dtype=np.float32))
        self.assertIs(warnings.warn, before)
        self.assertEqual(int(labels.max()), 0)
        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
