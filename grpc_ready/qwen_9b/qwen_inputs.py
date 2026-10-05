"""Small, model-specific input files shared by the artifact entry points."""

import json
from pathlib import Path
import re


def validate_input_records(records, model_profile):
    formal, warmup = [], []
    seen = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        if record.get("model_profile") != model_profile:
            raise ValueError(f"Record {index}: prepare inputs for --model {model_profile}.")
        ids = record.get("input_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError(f"Record {index}: input_ids must be non-empty non-negative integers.")
        for field in ("record_id", "dataset"):
            value = record.get(field)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError(f"Record {index}: {field} must use letters, digits, '_' or '-'.")
        if record["record_id"] in seen:
            raise ValueError(f"Duplicate record_id: {record['record_id']}")
        seen.add(record["record_id"])
        if record.get("input_group") not in {"short", "long"}:
            raise ValueError(f"Record {index}: input_group must be short or long.")
        phase = record.get("phase")
        if phase not in {"formal", "warmup"}:
            raise ValueError(f"Record {index}: phase must be formal or warmup.")
        (warmup if phase == "warmup" else formal).append(record)
    if not formal:
        raise ValueError("The input file must contain at least one formal record.")
    return formal, warmup


def load_input_file(path, model_profile):
    with Path(path).expanduser().open(encoding="utf-8") as input_file:
        records = [json.loads(line) for line in input_file if line.strip()]
    return validate_input_records(records, model_profile)
