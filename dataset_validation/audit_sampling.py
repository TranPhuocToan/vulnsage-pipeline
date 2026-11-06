"""Deterministic stratified sampling helpers for independent dataset audits."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence


def stratified_sample(
    records: Sequence[Mapping[str, Any]],
    quotas: Mapping[tuple[str, str], int],
    *,
    seed: int,
    excluded_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = excluded_source_ids or set()
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(quotas):
        status, project = stratum
        frame = sorted(
            (
                dict(record)
                for record in records
                if record.get("status") == status
                and record.get("project") == project
                and str(record.get("source_id")) not in excluded
            ),
            key=lambda record: int(record["source_id"]),
        )
        quota = quotas[stratum]
        if quota < 0 or quota > len(frame):
            raise ValueError(f"Invalid quota {quota} for stratum {stratum}")
        selected.extend(rng.sample(frame, quota))
    return selected
