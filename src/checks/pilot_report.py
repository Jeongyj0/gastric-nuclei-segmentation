import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
from uuid import uuid4
from src.study import TISSUES, CONDITIONS, POLICY, project_path, json_hash, write_json
from src.train.control import validation_due, learning_rate_after
from src.train.layout import run_layout
from src.train.pilot import load_pilot, select_targets


def analyze_rows(rows, settings):
    maximum, interval = settings["max_optimizer_updates"], settings["validation_interval_updates"]
    if not rows:
        return [], None
    points = []
    last_update = 0
    for row in rows:
        try:
            update = int(row["update"])
            point = {key: float(row[key]) for key in (
                "val_mean_pq", "train_seconds", "validation_seconds", "checkpoint_seconds",
                "elapsed_seconds", "lr_used", "lr_next")}
        except (KeyError, TypeError, ValueError):
            raise ValueError("Pilot log has missing or invalid numeric fields.") from None
        expected = min(last_update + interval, maximum)
        if update != expected or update <= last_update or not validation_due(update, maximum, interval):
            raise ValueError("Validation update order, interval, or budget does not match the log.")
        if not all(math.isfinite(value) and value >= 0 for value in point.values()):
            raise ValueError("Pilot log contains nonfinite or negative values.")
        if point["val_mean_pq"] > 1 or point["lr_used"] <= 0 or point["lr_next"] <= 0:
            raise ValueError("PQ or LR values are out of range.")
        for key, completed in (("lr_used", update - 1), ("lr_next", update)):
            expected_lr = learning_rate_after(completed, settings["lr_schedule"])
            if not math.isclose(point[key], expected_lr, rel_tol=1e-8, abs_tol=1e-15):
                raise ValueError("Logged LR schedule does not match the saved settings.")
        if points and any(point[key] < points[-1][key] for key in ("train_seconds", "elapsed_seconds")):
            raise ValueError("Cumulative runtime decreased.")
        if point["elapsed_seconds"] < point["train_seconds"]:
            raise ValueError("Elapsed time is shorter than training time.")
        point["update"] = update
        points.append(point)
        last_update = update
    best = max(points, key=lambda value: value["val_mean_pq"])
    last = points[-1]
    recent = [point["val_mean_pq"] for point in points[-3:]]
    earlier = [point["val_mean_pq"] for point in points[:-3]]
    metrics = {
        "logged_validations": len(points), "last_update": last["update"],
        "best_update": best["update"], "best_validation_pq": best["val_mean_pq"],
        "last_validation_pq": last["val_mean_pq"],
        "recent_three_pq_range": max(recent) - min(recent) if len(recent) == 3 else None,
        "recent_best_minus_earlier_best": max(recent) - max(earlier) if earlier else None,
        "lower_lr_validation_points": sum(p["lr_used"] < settings["lr_schedule"]["initial_lr"] for p in points),
        "train_seconds": last["train_seconds"],
        "train_seconds_per_update": last["train_seconds"] / last["update"],
        "median_validation_seconds": statistics.median(p["validation_seconds"] for p in points),
        "median_checkpoint_seconds": statistics.median(p["checkpoint_seconds"] for p in points),
        "elapsed_seconds_at_last_validation": last["elapsed_seconds"],
        "automatic_budget_selection": False,
    }
    return points, metrics


def estimate_seconds(metrics, maximum, interval):
    if type(maximum) is not int or type(interval) is not int or not 0 < interval <= maximum:
        raise ValueError("Invalid update budget or validation interval for time projection.")
    evaluations = (maximum + interval - 1) // interval
    return (metrics.get("initialization_seconds", 0.0) + maximum * metrics["train_seconds_per_update"] +
            evaluations * (metrics["median_validation_seconds"] + metrics["median_checkpoint_seconds"]))


def collect_runs(document, settings):
    runs = []
    expected_pairs = select_targets(document)
    scope = run_layout(*expected_pairs[0], document["pilot_id"])["scope"]
    protocol = None
    for tissue, condition, seed in expected_pairs:
        folder = run_layout(tissue, condition, seed, document["pilot_id"])["out"]
        record = {"tissue": tissue, "condition": condition, "seed": seed, "state": "not_started",
                  "points": [], "metrics": None}
        runs.append(record)
        if not folder.exists():
            continue
        if protocol is None:
            protocol = json.loads(project_path(scope / "protocol.json", must_exist=True).read_text(encoding="utf-8"))
            if (protocol.get("phase") != "pilot" or protocol.get("pilot_id") != document["pilot_id"] or
                    protocol["settings"] != settings or protocol["policy"] != POLICY or
                    protocol["num_workers"] != document["num_workers"] or
                    protocol["inference_batch_size"] != document["inference_batch_size"]):
                raise ValueError("Pilot settings differ from the run protocol. Use the original settings.")
        config = json.loads(project_path(folder / "config.json", must_exist=True).read_text(encoding="utf-8"))
        status = json.loads(project_path(folder / "status.json", must_exist=True).read_text(encoding="utf-8"))
        for value in (config, status):
            if value.get("phase") != "pilot" or value.get("pilot_id") != document["pilot_id"]:
                raise ValueError("Run records include a different phase or pilot.")
        if ((config["tissue"], config["condition"], config["seed"]) != (tissue, condition, seed) or
                config["settings"] != settings or config["protocol_sha256"] != json_hash(protocol)):
            raise ValueError("Pilot run provenance or settings mismatch.")
        with project_path(folder / "log.csv", must_exist=True).open(encoding="utf-8", newline="") as stream:
            points, metrics = analyze_rows(list(csv.DictReader(stream)), settings)
        if status["status"] not in ("running", "completed", "failed"):
            raise ValueError("Unknown pilot run status.")
        if status["status"] == "completed":
            if not metrics or status["update"] != settings["max_optimizer_updates"] or metrics["last_update"] != status["update"]:
                raise ValueError("Completed status does not match the final validation update.")
            best = status["best"]["pq"]
            if (best["update"] != metrics["best_update"] or
                    not math.isclose(best["score"], metrics["best_validation_pq"], rel_tol=0, abs_tol=1e-12)):
                raise ValueError("Selected checkpoint record differs from the validation log.")
        if metrics:
            for key in ("initialization_seconds", "total_seconds"):
                value = status.get(key)
                if value is not None:
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError("Invalid completion or failure timing record.")
                    metrics[key] = value
        record.update(state=status["status"], points=points, metrics=metrics)
    return runs


def plot_curves(runs, out, settings):
    available = [run for run in runs if run["points"]]
    if not available:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(out / "plot_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = []
    colors = dict(zip(CONDITIONS, ("#1565c0", "#e67e22", "#2e7d32", "#9c27b0", "#d32f2f")))
    for tissue in TISSUES:
        selected = [run for run in available if run["tissue"] == tissue]
        if not selected:
            continue
        figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
        figure.suptitle(f"{tissue} | Train/validation pilot (not test results)")
        for run in selected:
            points = run["points"]
            x = [point["update"] for point in points]
            label = f"{run['condition']} / seed {run['seed']} / {run['state']}"
            color = colors[run["condition"]]
            axes[0].plot(x, [p["val_mean_pq"] for p in points], "o-", ms=4, color=color, label=label)
            axes[2].plot(x, [p["elapsed_seconds"] / 3600 for p in points], "o-", ms=4, color=color)
        maximum = max(run["points"][-1]["update"] for run in selected)
        step = settings["lr_schedule"]["step_size_updates"]
        lr_x = sorted({0, maximum, *range(step, maximum + 1, step)})
        axes[1].step(lr_x, [learning_rate_after(u, settings["lr_schedule"]) for u in lr_x],
                     where="post", color="#37474f")
        axes[0].set_ylabel("Validation image-mean PQ")
        axes[0].set_ylim(0, 1)
        axes[0].legend(fontsize=8, loc="best")
        axes[1].set_ylabel("LR for next update")
        axes[1].set_yscale("log")
        axes[2].set_ylabel("Cumulative loop time (hours)")
        axes[2].set_xlabel("Completed optimizer updates")
        for axis in axes:
            axis.grid(alpha=0.2)
        name = f"{tissue}_pq_lr_time.png"
        figure.savefig(out / name, dpi=160)
        plt.close(figure)
        figures.append(name)
    return figures


def write_report(document, settings, runs, out, *, candidate_updates=(), candidate_interval=None, main_seeds=None):
    out = project_path(out)
    if out.exists():
        raise FileExistsError("Report directory already exists. Specify a new output path.")
    interval = candidate_interval or settings["validation_interval_updates"]
    if main_seeds is not None and (type(main_seeds) is not int or main_seeds < 1):
        raise ValueError("Main-study seed count for time projection must be a positive integer.")
    for maximum in candidate_updates:
        if type(maximum) is not int or not 0 < interval <= maximum:
            raise ValueError("Invalid update budget or validation interval for time projection.")
    projections = []
    for maximum in candidate_updates:
        cases = {}
        for run in runs:
            if run["metrics"] and run["state"] == "completed":
                cases.setdefault((run["tissue"], run["condition"]), []).append(
                    estimate_seconds(run["metrics"], maximum, interval))
        case_hours = {f"{t}/{c}": statistics.median(values) / 3600 for (t, c), values in cases.items()}
        full_coverage = set(cases) == {(t, c) for t in TISSUES for c in CONDITIONS}
        projections.append({"candidate_updates": maximum, "validation_interval_updates": interval,
                            "observed_case_hours": case_hours, "covered_cases": len(cases),
                            "required_cases": len(TISSUES) * len(CONDITIONS), "main_seed_count": main_seeds,
                            "full_study_days": (sum(case_hours.values()) * main_seeds / 24
                                                if full_coverage and main_seeds is not None else None)})
    out.mkdir(parents=True, exist_ok=False)
    figures = plot_curves(runs, out, settings)
    summary = {"pilot_id": document["pilot_id"], "settings": settings, "runs": runs,
               "time_projections": projections, "figures": figures,
               "test_inference_executed": False, "automatic_budget_selection": False,
               "scope": "Train/validation aggregate logs; no automatic convergence or main-study budget selection"}
    write_json(out / "pilot_summary.json", summary)
    flat = [{"tissue": run["tissue"], "condition": run["condition"], "seed": run["seed"],
             "state": run["state"], **(run["metrics"] or {})} for run in runs]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (out / "pilot_summary.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Train/validation pilot analysis", "", f"Pilot: `{document['pilot_id']}`", "",
             "This report analyzes validation summaries only; it does not automatically select the main-study budget or LR from PQ.", "",
             "| Tissue | Condition | Seed | Status | Last update | Best PQ | Best update | Last PQ | Logged hours |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    names = {"not_started": "not started", "running": "running", "completed": "completed", "failed": "failed"}
    for run in runs:
        prefix = f"| {run['tissue']} | {run['condition']} | {run['seed']} | {names[run['state']]} |"
        m = run["metrics"]
        if m:
            hours = m.get("total_seconds", m["elapsed_seconds_at_last_validation"]) / 3600
            lines.append(prefix + f" {m['last_update']} | {m['best_validation_pq']:.5f} | {m['best_update']} | {m['last_validation_pq']:.5f} | {hours:.2f} |")
        else:
            lines.append(prefix + " — | — | — | — | — |")
    lines += ["", "## Curves", ""]
    if not figures:
        lines.append("No curves were generated because validation records are unavailable.")
    for name in figures:
        lines.append(f"![Validation PQ, LR, and runtime]({name})")
    lines += ["", "## Budget review", "",
              "- Check whether PQ still improves near the last observation or after LR reduction. A short plateau alone does not establish convergence.",
              "- `recent_best_minus_earlier_best` subtracts the earlier best PQ from the best of the last 3 evaluations. It is descriptive, not a convergence threshold or significance test.",
              "- Best-PQ updates may differ by condition. Main-study runs use a common maximum update budget and select their best validation PQ.",
              "- Interpret only the observed tissues, conditions, and seeds. A Base-only pilot cannot establish convergence for synthetic-data conditions; use the results to plan further checks.",
              "- Unstarted or failed runs are incomplete. Running logs reflect only the time they were read.",
              "- Fix the main-study budget, LR, batch size, and validation interval after the pilot, then restart from shared pretrained weights. Do not select settings using test performance.",
              "", "## Timing and projections", "",
              "`train_seconds` is cumulative training time including data loading; validation/checkpoint times are per evaluation. For completed or failed runs, total_seconds spans model setup through termination, excluding pre-run manifest checks and Python startup.",
              "Projected time = initialization + candidate updates * observed training seconds/update + evaluation count * (median validation time + median checkpoint time). Candidates come from the analysis command and are not saved as study settings.",
              "Actual time varies with training, storage, system load, and other I/O costs. Full-study time requires observations for all 20 tissue/condition combinations."]
    if not projections:
        lines += ["", "No projection candidates were specified. Review observed curves and timing first."]
    for item in projections:
        lines += ["", f"### Candidate: {item['candidate_updates']:,} updates / validation interval {interval:,}", "",
                  f"Coverage: {item['covered_cases']}/20 combinations."]
        for case, hours in item["observed_case_hours"].items():
            lines.append(f"- {case}: approximately {hours:.2f} hours per run")
        if item["full_study_days"] is None:
            lines.append("Full-study time unavailable: all combinations and a main-study seed count are required.")
        else:
            lines.append(f"Full-study projection: {item['full_study_days']:.2f} continuous days; this is an estimate.")
    (out / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze pilot validation logs and project runtime.")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--candidate-updates", type=int, nargs="+", default=[])
    parser.add_argument("--candidate-validation-interval", type=int)
    parser.add_argument("--main-seeds", type=int)
    args = parser.parse_args(argv)
    document, _, settings = load_pilot(args.config)
    if args.candidate_validation_interval is not None and args.candidate_validation_interval < 1:
        parser.error("Validation interval must be a positive integer.")
    runs = collect_runs(document, settings)
    stamp = uuid4().hex[:8]
    out = args.output_dir or f"outputs/pilot_reports/{document['pilot_id']}/{stamp}"
    summary = write_report(document, settings, runs, out, candidate_updates=args.candidate_updates,
                           candidate_interval=args.candidate_validation_interval, main_seeds=args.main_seeds)
    print(f"Analyzed runs: {sum(bool(run['points']) for run in runs)}/{len(runs)}; no training or test inference")
    print("Report:", project_path(out).relative_to(project_path(".")))
    print("Shared main-study settings are unchanged.")
    return summary


if __name__ == "__main__":
    main()
