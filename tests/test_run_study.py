import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config_fixture import read_study_config
from src.train import run_study


class StudyRunnerTests(unittest.TestCase):
    def test_direct_run_creates_and_releases_lock_in_fresh_project(self):
        config = read_study_config()
        with tempfile.TemporaryDirectory(prefix="study-run-check-") as temporary:
            root = Path(temporary).resolve()

            def check_subprocess(command, *, cwd, check):
                self.assertTrue((root / "runs/study.lock").is_file())
                self.assertEqual(cwd, root)
                self.assertTrue(check)
                self.assertIn("src.train.train", command)

            with patch("src.study.ROOT", root), \
                 patch.object(run_study, "ROOT", root), \
                 patch.object(run_study, "load_config", return_value=config), \
                 patch.object(run_study.sys, "argv", ["run_study", "--run"]), \
                 patch.object(run_study.subprocess, "run", side_effect=check_subprocess) as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                run_study.main()
            self.assertEqual(launch.call_count, 20)
            self.assertFalse((root / "runs/study.lock").exists())
            status = json.loads((root / "runs/study_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_runs"], 20)


if __name__ == "__main__":
    unittest.main()
