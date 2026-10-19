import argparse
import ast
import io
import os
import sys
import traceback
import unittest
from uuid import uuid4
from src.study import ROOT, load_config, training_settings, audit_manifests, write_json, project_path


def main():
    parser = argparse.ArgumentParser(description="Check study inputs, settings, and runtime readiness.")
    configuration = parser.add_mutually_exclusive_group()
    configuration.add_argument("--config", default="configs/study.json")
    configuration.add_argument("--pilot-config", help="Check pilot settings separately from main-study settings")
    parser.add_argument("--runtime", action="store_true", help="Check model forward pass and loss without training")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", help="New directory for check results; existing directories are not overwritten")
    args = parser.parse_args()
    out = (project_path(args.output_dir) if args.output_dir else
           ROOT / "outputs" / ("checks_" + uuid4().hex[:8]))
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(out / "runtime_cache/matplotlib"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(out / "runtime_cache/cuda"))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    result = {"training_executed": False,
              "readiness_scope": "pilot" if args.pilot_config else "main",
              "python": sys.executable, "runtime_requested": args.runtime}
    try:
        paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "source_core").rglob("*.py"))
        for path in paths:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        result["parsed_sources"] = len(paths)
        if args.pilot_config:
            from src.train.pilot import load_pilot
            document, config, _ = load_pilot(args.pilot_config)
            result["pilot_id"] = document["pilot_id"]
        else:
            config = load_config(args.config)
        result["manifests"] = audit_manifests(config)
        try:
            result["training_settings"] = training_settings(config)
            result["training_settings_complete"] = True
        except ValueError as error:
            result["training_settings_complete"] = False
            result["settings_message"] = str(error)
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        output = io.StringIO()
        checks = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        (out / "unit_checks.txt").write_text(output.getvalue(), encoding="utf-8")
        result["unit_checks"] = {"run": checks.testsRun, "failures": len(checks.failures),
                                 "errors": len(checks.errors), "passed": checks.wasSuccessful()}
        if not checks.wasSuccessful():
            raise RuntimeError("Regression checks failed")
        result["code_checks_passed"] = True
        if args.runtime:
            try:
                from src.checks.runtime import check_runtime
                result["runtime"] = check_runtime(args.device)
                result["runtime_passed"] = True
            except Exception as error:
                result["runtime_passed"] = False
                result["runtime_error_type"] = type(error).__name__
                (out / "runtime_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    except Exception as error:
        result["code_checks_passed"] = False
        result["error_type"] = type(error).__name__
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    result["ready_to_train"] = (result.get("code_checks_passed", False) and
                                result.get("training_settings_complete", False) and
                                result.get("runtime_passed", False) and args.device == "cuda")
    result["ready_for_main_training"] = result["ready_to_train"] and result["readiness_scope"] == "main"
    write_json(out / "preflight.json", result)
    print("Code/manifest checks:", "passed" if result.get("code_checks_passed") else "failed")
    print("Shared training settings:", "complete" if result.get("training_settings_complete") else "missing values")
    print("Settings checked:", "pilot" if args.pilot_config else "main study")
    print("PyTorch check:", ("passed" if result.get("runtime_passed") else "failed/unavailable")
          if args.runtime else "not requested")
    print("Check results:", out.name)
    return 0 if result["ready_to_train"] or (not args.runtime and result.get("code_checks_passed")) else 2


if __name__ == "__main__":
    sys.exit(main())
