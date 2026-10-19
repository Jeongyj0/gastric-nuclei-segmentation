from pathlib import Path
import re
from src.study import TISSUES, CONDITIONS, project_path


def run_layout(tissue, condition, seed, pilot_id=None):
    if tissue not in TISSUES or condition not in CONDITIONS:
        raise ValueError("Unknown tissue or condition.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed is out of range.")
    if pilot_id is None:
        relative = Path("runs")
        phase = "main"
    else:
        if (not isinstance(pilot_id, str) or
                not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", pilot_id) or
                pilot_id in {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1, 10)],
                             *[f"lpt{i}" for i in range(1, 10)]}):
            raise ValueError("pilot_id must start with a lowercase letter and contain at most 48 characters.")
        relative = Path("runs/pilots") / pilot_id
        phase = "pilot"
    scope = project_path(relative)
    return {"scope": scope, "out": project_path(relative / "models" / tissue / condition / f"seed_{seed}"),
            "phase": phase, "pilot_id": pilot_id}


def require_main_run(record):
    if record.get("phase", "main") != "main" or record.get("pilot_id") is not None:
        raise ValueError("Pilot checkpoints cannot be used for main test evaluation.")
