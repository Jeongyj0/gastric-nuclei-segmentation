import contextlib
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import start_training


class LauncherTests(unittest.TestCase):
    def run_case(self, batch_passed=True, memory=2048, preflight_passed=True, fail_command=None):
        calls = []
        with tempfile.TemporaryDirectory(prefix="study-launch-check-") as temporary:
            root = Path(temporary)
            attempt = root / "outputs/attempt_1"
            attempt.mkdir(parents=True)

            def fake_run(command, log_path):
                self.assertEqual(command[0], sys.executable)
                module = command[command.index("-m") + 1]
                calls.append(module)
                if module == fail_command:
                    raise RuntimeError("Test subprocess failed.")
                if module == "src.checks.batch_probe":
                    self.assertEqual(command[command.index("--num-workers") + 1], "4")
                    output = attempt / "batch_check/result.json"
                    result = dict(passed=batch_passed, batch_size=2, free_device_mib_after_probe=memory)
                elif module == "src.checks.preflight":
                    output = attempt / "preflight/preflight.json"
                    result = dict(ready_for_main_training=preflight_passed)
                else:
                    self.assertIn("--run", command)
                    self.assertEqual(command[command.index("--num-workers") + 1], "4")
                    return
                output.parent.mkdir()
                output.write_text(json.dumps(result), encoding="utf-8")

            with patch.object(start_training, "ROOT", root), \
                 patch.object(start_training, "run_logged", side_effect=fake_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                if not batch_passed or memory < 512 or not preflight_passed or fail_command:
                    with self.assertRaises(RuntimeError):
                        start_training.run_checks_and_training(attempt)
                else:
                    start_training.run_checks_and_training(attempt)
        return calls

    def test_all_checks_must_pass_before_training(self):
        self.assertEqual(self.run_case(), ["src.checks.batch_probe", "src.checks.preflight", "src.train.run_study"])

    def test_failed_worker_check_never_starts_training(self):
        self.assertEqual(self.run_case(batch_passed=False), ["src.checks.batch_probe"])

    def test_insufficient_memory_never_starts_training(self):
        self.assertEqual(self.run_case(memory=0), ["src.checks.batch_probe"])

    def test_failed_preflight_never_starts_training(self):
        self.assertEqual(self.run_case(preflight_passed=False), ["src.checks.batch_probe", "src.checks.preflight"])

    def test_nonzero_check_command_stops_the_sequence(self):
        self.assertEqual(self.run_case(fail_command="src.checks.batch_probe"), ["src.checks.batch_probe"])

    def test_fresh_project_creates_runs_and_cleans_launch_lock(self):
        config = {"batch_size": 2, "main_execution": {"num_workers": 4}}
        with tempfile.TemporaryDirectory(prefix="study-launch-check-") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/study.json").write_text(json.dumps(config), encoding="utf-8")

            def check_launch(attempt):
                self.assertTrue((root / "runs/launch.lock").is_file())
                self.assertEqual(attempt, root / "outputs/training/attempt_1")

            with patch.object(start_training, "ROOT", root), \
                 patch.object(start_training, "load_config", return_value=config), \
                 patch.object(start_training, "training_settings"), \
                 patch.object(start_training, "run_checks_and_training", side_effect=check_launch) as launch:
                self.assertEqual(start_training.main([]), 0)
            launch.assert_called_once()
            self.assertTrue((root / "runs").is_dir())
            self.assertFalse((root / "runs/launch.lock").exists())

    def test_plan_uses_active_interpreter_without_creating_runs(self):
        with tempfile.TemporaryDirectory(prefix="study-launch-check-") as temporary:
            root = Path(temporary)
            with patch.object(start_training, "ROOT", root), \
                 patch.object(start_training, "load_config", return_value={}), \
                 patch.object(start_training, "training_settings"), \
                 patch.object(start_training.subprocess, "run") as command, \
                 patch.object(start_training, "run_checks_and_training") as launch:
                self.assertEqual(start_training.main(["--plan-only"]), 0)
            self.assertEqual(command.call_args.args[0][0], sys.executable)
            self.assertNotIn("--run", command.call_args.args[0])
            launch.assert_not_called()
            self.assertFalse((root / "runs").exists())


if __name__ == "__main__":
    unittest.main()
