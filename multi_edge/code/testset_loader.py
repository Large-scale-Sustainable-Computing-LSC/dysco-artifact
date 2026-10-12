"""Loads raw-text rows from workshop/selected_testsets for grpc_multi benches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "selected_testsets"


def load_prompts(
    group: str,
    root: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[str]:
    base = Path(root) if root is not None else DEFAULT_ROOT
    path = base / group / "raw_inputs.jsonl"
    prompts: List[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row: Dict[str, Any] = json.loads(line)
            prompts.append(row["input_text"])
            if limit is not None and len(prompts) >= limit:
                break
    return prompts
