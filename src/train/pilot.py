import argparse
import copy
import json
import subprocess
import sys
from types import SimpleNamespace
from src.study import (ROOT, TISSUES, CONDITIONS, project_path, load_config,
                       training_settings, audit_manifests, execution_policy)
from src.train.layout import run_layout

EXECUTION_KEYS = ("training_seeds", "batch_size", "max_optimizer_updates", "lr_schedule",
                  "validation_interval_updates", "encoder_freeze_schedule")


def load_pilot(path="configs/pilot.json"):
    document = json.loads(project_path(path, must_exist=True).read_text(encoding="utf-8-sig"))
    if document.get("purpose") != "train_validation_budget_selection":
        raise ValueError("Pilot purpose must be training/validation budget review.")
    base = load_config(document["study_config"])
    config = copy.deepcopy(base)
    for key in EXECUTION_KEYS:
        config[key] = document.get(key)
    config.pop("early_stopping", None)
    config["policy"] = execution_policy(config)
    settings = training_settings(config)
    for key, allowed in (("tissues", TISSUES), ("conditions", CONDITIONS)):
        items = document.get(key)
        if (not isinstance(items, list) or not items or
                any(not isinstance(item, str) or item not in allowed for item in items) or
                len(set(items)) != len(items)):
            raise ValueError("Invalid pilot tissue or condition list.")
    for key, minimum in (("num_workers", 0), ("inference_batch_size", 1)):
        if type(document.get(key)) is not int or document[key] < minimum:
            raise ValueError("Invalid pilot worker count or inference batch size.")
    run_layout(document["tissues"][0], document["conditions"][0], settings["training_seeds"][0],
               document.get("pilot_id"))
    if document.get("pilot_id") is None:
        raise ValueError("Pilot name is required.")
    if config["model_mode"] != "fast" or config["nuclear_type_classification"]:
        raise ValueError("Pilot runs must use the same NP/HV model.")
    return document, config, settings


def select_targets(document, *, tissue=None, condition=None, seed=None):
    for value, key in ((tissue, "tissues"), (condition, "conditions"), (seed, "training_seeds")):
        if value is not None and value not in document[key]:
            raise ValueError("Requested run is not included in the pilot configuration.")
    return [(t, c, s) for s in document["training_seeds"] for t in document["tissues"]
            for c in document["conditions"]
            if (tissue is None or t == tissue) and (condition is None or c == condition)
            and (seed is None or s == seed)]


def command_for(config_path, target):
    tissue, condition, seed = target
    return [sys.executable, "-X", "utf8", "-B", "-m", "src.train.pilot", "--config", config_path,
            "--cancer", tissue, "--condition", condition, "--seed", str(seed), "--run"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plan or run pilot training.")
    parser.add_argument("--config", default="configs/pilot.json")
    parser.add_argument("--cancer", choices=TISSUES)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run", action="store_true", help="Start pilot training")
    action.add_argument("--check", action="store_true", help="Check frozen manifests without training")
    args = parser.parse_args(argv)
    try:
        document, config, settings = load_pilot(args.config)
        targets = select_targets(document, tissue=args.cancer, condition=args.condition, seed=args.seed)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Pilot {document['pilot_id']}: {len(targets)} runs, {settings['max_optimizer_updates']} updates each")
    print("Separate from main configuration; no test inference or automatic convergence assessment.")
    for target in targets:
        print(subprocess.list2cmdline(command_for(args.config, target)))
    if not args.run and not args.check:
        print("Plan printed. Training not started.")
        return
    audit = audit_manifests(config)
    if args.check:
        print("Frozen manifest and patient separation checks passed. Training not started.")
        return
    for tissue, condition, seed in targets:
        if run_layout(tissue, condition, seed, document["pilot_id"])["out"].exists():
            raise FileExistsError("Pilot run already exists. Select an unrun condition or a new pilot_id.")
    if len(targets) == 1:
        from src.train.train import run
        tissue, condition, seed = targets[0]
        run_args = SimpleNamespace(cancer=tissue, condition=condition, seed=seed,
                                   num_workers=document["num_workers"],
                                   inference_batch_size=document["inference_batch_size"])
        run(run_args, config, settings, audit, pilot_id=document["pilot_id"])
    else:
        for target in targets:
            subprocess.run(command_for(args.config, target), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
