import json
from pathlib import Path


def read_study_config():
    path = Path(__file__).resolve().parents[1] / "configs/study.json"
    return json.loads(path.read_text(encoding="utf-8-sig"))
