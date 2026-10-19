import copy
from types import SimpleNamespace
import unittest
from src.study import training_settings
from config_fixture import read_study_config
from src.train.control import ValidationPlateau, select_checkpoints, validate_completed_run


def controller():
    optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}])
    lr = {"kind": "plateau_once", "initial_lr": 1e-4, "reduced_lr": 1e-5, "patience_checks": 3}
    early = {"min_delta": 0.001, "patience_checks": 5, "start_after_updates": 3000}
    return ValidationPlateau(optimizer, lr, early)


class EarlyStoppingTests(unittest.TestCase):
    def test_warmup_then_third_bad_reduces_fifth_bad_stops(self):
        control = controller()
        for epoch in range(1, 6):
            self.assertFalse(control.step(0.4, epoch * 600))
            self.assertEqual(control.bad_checks, 0)
        for epoch in range(6, 11):
            self.assertEqual(control.step(0.4, epoch * 600), epoch == 10)
            self.assertEqual(control.bad_checks, epoch - 5)
            self.assertEqual(control.optimizer.param_groups[0]["lr"], 1e-4 if epoch < 8 else 1e-5)
        self.assertEqual(control.reduced_at_update, 4800)

    def test_small_increases_save_true_best_and_accumulate_for_significant_improvement(self):
        control = controller()
        best = {"pq": None, "np_dice": None}
        for update, pq in ((3000, 0.4), (3600, 0.4006), (4200, 0.4012)):
            self.assertIn("best_pq.pth", select_checkpoints(best, pq, 0.7, update))
            control.step(pq, update)
            self.assertEqual(control.bad_checks, 1 if update == 3600 else 0)
        self.assertEqual(best["pq"]["score"], 0.4012)
        self.assertEqual(control.reference_pq, 0.4012)
        self.assertNotIn("best_pq.pth", select_checkpoints(best, 0.4012, 0.7, 4800))
        self.assertEqual(best["pq"]["update"], 4200)

    def test_improvement_after_reduction_resets_counter_without_second_lr_reduction(self):
        control = controller()
        for update in (3000, 3600, 4200, 4800):
            control.step(0.4, update)
        self.assertTrue(control.lr_reduced)
        control.step(0.402, 5400)
        self.assertEqual(control.bad_checks, 0)
        for index, update in enumerate(range(6000, 8401, 600), 1):
            self.assertEqual(control.step(0.402, update), index == 5)
            self.assertEqual(control.optimizer.param_groups[0]["lr"], 1e-5)
        self.assertEqual(control.reduced_at_update, 4800)

    def test_exact_delta_is_not_improvement_and_warmup_keeps_best(self):
        control = controller()
        control.step(0.5, 600)
        control.step(0.4, 3000)
        control.step(0.501, 3600)
        self.assertEqual(control.reference_pq, 0.5)
        self.assertEqual(control.bad_checks, 1)

    def test_nonfinite_out_of_order_and_post_stop_validation_rejected(self):
        for pq in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                controller().step(pq, 600)
        control = controller()
        for update in range(3000, 6001, 600):
            control.step(0.4, update)
        for update in (6000, 6600):
            with self.assertRaises(ValueError):
                control.step(0.6, update)

    def test_early_completed_accepted_but_partial_and_unjustified_completion_rejected(self):
        control = controller()
        for update in range(3000, 6001, 600):
            control.step(0.4, update)
        settings = {"max_optimizer_updates": 30000, "validation_interval_updates": 600,
                    "lr_schedule": control.lr_schedule, "early_stopping": control.early_stopping}
        status = {"status": "completed", "update": 6000, "stop_reason": "early_stopping",
                  "validation_control": control.state_dict()}
        validate_completed_run(status, settings)
        for key, value in (("status", "failed"), ("stop_reason", "max_optimizer_updates"), ("update", 5999)):
            with self.assertRaises(ValueError):
                validate_completed_run({**status, key: value}, settings)
        bad = copy.deepcopy(status)
        bad["validation_control"]["bad_checks"] = 4
        with self.assertRaises(ValueError):
            validate_completed_run(bad, settings)
        validate_completed_run({"status": "completed", "update": 30000}, settings)

    def test_configuration_rejects_inconsistent_controls(self):
        config = read_study_config()
        training_settings(config)
        for field, value in (("min_delta", float("nan")), ("patience_checks", 3),
                             ("start_after_updates", 1)):
            bad = copy.deepcopy(config)
            bad["early_stopping"][field] = value
            with self.assertRaises(ValueError):
                training_settings(bad)


if __name__ == "__main__":
    unittest.main()
