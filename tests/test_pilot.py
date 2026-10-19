import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.study import training_settings, POLICY, json_hash
from config_fixture import read_study_config
from src.train.layout import run_layout, require_main_run
from src.train.pilot import select_targets, main as pilot_main
from src.checks.pilot_report import analyze_rows, estimate_seconds, collect_runs, write_report


def configuration():
    config = read_study_config()
    config.update(training_seeds=[42], batch_size=1, max_optimizer_updates=11,
                  validation_interval_updates=4,
                  lr_schedule={"kind": "step", "initial_lr": 0.0001, "step_size_updates": 6, "gamma": 0.1},
                  encoder_freeze_schedule={"freeze_updates": 0})
    config.pop("early_stopping", None)
    config["policy"] = copy.deepcopy(POLICY)
    document = {"pilot_id": "testprobe", "tissues": ["STNT"], "conditions": ["base", "add"],
                "training_seeds": [42], "num_workers": 4, "inference_batch_size": 1}
    return document, config, training_settings(config)


def rows():
    return [dict(update=u, val_mean_pq=pq, train_seconds=u*2, validation_seconds=5,
                 checkpoint_seconds=1, elapsed_seconds=u*2+6*(i+1),
                 lr_used=(0.0001 if u <= 6 else 0.00001),
                 lr_next=(0.0001 if u < 6 else 0.00001))
            for i, (u, pq) in enumerate(((4, 0.4), (8, 0.7), (11, 0.7)))]


class PilotIsolationTests(unittest.TestCase):
    def test_main_and_pilot_outputs_and_locks_are_disjoint(self):
        main = run_layout("STNT", "base", 42)
        pilot = run_layout("STNT", "base", 42, "probe01")
        self.assertNotEqual(main["out"], pilot["out"])
        self.assertNotEqual(main["scope"] / "protocol.json", pilot["scope"] / "protocol.json")
        self.assertNotEqual(main["scope"] / "initialization", pilot["scope"] / "initialization")
        self.assertTrue(pilot["out"].is_relative_to(main["scope"] / "pilots"))

    def test_invalid_names_cannot_escape_scope(self):
        for name in ("../main", "D:/outside", "../", "CON", "nul", "a/b", "", "a"*49):
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_layout("STNT", "base", 42, name)

    def test_pilot_records_and_checkpoints_are_not_main(self):
        for record in ({"phase": "pilot"}, {"pilot_id": "probe01"}, {"phase": "unknown"}):
            with self.assertRaises(ValueError):
                require_main_run(record)
        require_main_run({"phase": "main", "pilot_id": None})
        require_main_run({})

    def test_default_command_never_imports_training_or_audits_data(self):
        doc, config, settings = configuration()
        with patch("src.train.pilot.load_pilot", return_value=(doc, config, settings)), \
             patch("src.train.pilot.audit_manifests") as audit, \
             patch("src.train.pilot.subprocess.run") as spawn, \
             patch("src.train.train.run") as train, contextlib.redirect_stdout(io.StringIO()):
            pilot_main([])
        audit.assert_not_called()
        spawn.assert_not_called()
        train.assert_not_called()

    def test_explicit_single_run_passes_pilot_scope_to_shared_trainer(self):
        doc, config, settings = configuration()
        with patch("src.train.pilot.load_pilot", return_value=(doc, config, settings)), \
             patch("src.train.pilot.audit_manifests", return_value={}), \
             patch("src.train.pilot.run_layout", return_value={"out": Path("absent-pilot-unit-check")}), \
             patch("src.train.train.run") as train, contextlib.redirect_stdout(io.StringIO()):
            pilot_main(["--condition", "base", "--run"])
        self.assertEqual(train.call_count, 1)
        self.assertEqual(train.call_args.kwargs, {"pilot_id": "testprobe"})

    def test_check_only_never_trains(self):
        doc, config, settings = configuration()
        with patch("src.train.pilot.load_pilot", return_value=(doc, config, settings)), \
             patch("src.train.pilot.audit_manifests", return_value={}) as audit, \
             patch("src.train.train.run") as train, contextlib.redirect_stdout(io.StringIO()):
            pilot_main(["--check"])
        audit.assert_called_once()
        train.assert_not_called()

    def test_targets_stay_inside_declared_pilot(self):
        doc, _, _ = configuration()
        self.assertEqual(select_targets(doc, condition="add"), [("STNT", "add", 42)])
        for kwargs in ({"condition": "replace50"}, {"tissue": "STDI"}, {"seed": 43}):
            with self.assertRaises(ValueError):
                select_targets(doc, **kwargs)


class PilotReportTests(unittest.TestCase):
    def test_best_checkpoint_tie_is_earliest_not_last(self):
        _, _, settings = configuration()
        points, metrics = analyze_rows(rows(), settings)
        self.assertEqual(metrics["best_update"], 8)
        self.assertEqual(metrics["last_update"], 11)
        self.assertEqual(metrics["lower_lr_validation_points"], 2)
        self.assertFalse(metrics["automatic_budget_selection"])

    def test_nan_missing_reordered_and_wrong_lr_logs_fail(self):
        _, _, settings = configuration()
        variants = []
        for field, value in (("val_mean_pq", float("nan")), ("val_mean_pq", 1.1),
                             ("train_seconds", -1), ("lr_used", 0.0002)):
            bad = rows()
            bad[1][field] = value
            variants.append(bad)
        variants.extend((list(reversed(rows())), rows()[1:]))
        missing = rows()
        del missing[0]["checkpoint_seconds"]
        variants.append(missing)
        for bad in variants:
            with self.assertRaises(ValueError):
                analyze_rows(bad, settings)

    def test_partial_run_kept_and_no_validation_is_not_zero_pq(self):
        _, _, settings = configuration()
        self.assertIsNone(analyze_rows([], settings)[1])
        _, metrics = analyze_rows(rows()[:1], settings)
        self.assertEqual(metrics["last_update"], 4)
        self.assertIsNone(metrics["recent_three_pq_range"])

    def test_time_estimate_counts_final_nonmultiple_validation(self):
        metrics = dict(initialization_seconds=10, train_seconds_per_update=2,
                       median_validation_seconds=5, median_checkpoint_seconds=1)
        self.assertEqual(estimate_seconds(metrics, 11, 4), 50)

    def test_report_runs_without_test_files_and_does_not_extrapolate_missing_cases(self):
        document, _, settings = configuration()
        with tempfile.TemporaryDirectory(prefix="hovernet-pilot-unit-") as temporary:
            root = Path(temporary).resolve()
            with patch("src.study.ROOT", root):
                location = run_layout("STNT", "base", 42, document["pilot_id"])
                folder = location["out"]
                folder.mkdir(parents=True)
                protocol = {"phase": "pilot", "pilot_id": document["pilot_id"], "settings": settings,
                            "policy": POLICY, "num_workers": 4, "inference_batch_size": 1}
                config = {"phase": "pilot", "pilot_id": document["pilot_id"], "tissue": "STNT",
                          "condition": "base", "seed": 42, "settings": settings,
                          "protocol_sha256": json_hash(protocol)}
                status = {"phase": "pilot", "pilot_id": document["pilot_id"], "status": "completed",
                          "update": 11, "best": {"pq": {"update": 8, "score": 0.7}},
                          "initialization_seconds": 3, "total_seconds": 46}
                for path, value in ((location["scope"] / "protocol.json", protocol),
                                    (folder / "config.json", config), (folder / "status.json", status)):
                    path.write_text(json.dumps(value), encoding="utf-8")
                with (folder / "log.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows()[0]))
                    writer.writeheader()
                    writer.writerows(rows())
                collected = collect_runs(document, settings)
                self.assertEqual([r["state"] for r in collected], ["completed", "not_started"])
                with patch("src.checks.pilot_report.plot_curves", return_value=[]):
                    result = write_report(document, settings, collected, "outputs/report",
                                          candidate_updates=[20], candidate_interval=4, main_seeds=3)
                self.assertIsNone(result["time_projections"][0]["full_study_days"])
                self.assertFalse(result["test_inference_executed"])
                self.assertFalse((root / "configs/study.json").exists())
                self.assertFalse((root / "runs/protocol.json").exists())
                status["best"]["pq"]["update"] = 11
                (folder / "status.json").write_text(json.dumps(status), encoding="utf-8")
                with self.assertRaises(ValueError):
                    collect_runs(document, settings)


if __name__ == "__main__":
    unittest.main()
