"""Offline-only Phase 3C.2 preflight for canonical P0/P1/P2 runs.

The module does not construct a network transport.  A mock client factory must
be injected by the caller.  Qualification records remain separate from model
input and are used only to gate official execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pipeline.canonical_adapter import dataset_sha256
from pipeline.context_policy import ContextPolicy
from pipeline.evaluator import evaluate_positive_only_ablation
from pipeline.llm_client import StructuredLLMClient
from pipeline.offline_integration import prompt_hashes, run_canonical_sample
from pipeline.run_state import RunStateError, RunWorkspace, build_run_spec, hash_files, hash_value


PREFLIGHT_VERSION = "phase3c2-offline-preflight-1.0"
PIPELINES = ("P0", "P1", "P2")


class PreflightError(ValueError):
    pass


def _atomic_create_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PreflightError(f"refusing to overwrite artifact: {path}")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise PreflightError(f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_dataset_manifest(
    *,
    dataset_path: Path,
    adapted_samples: Sequence[Mapping[str, Any]],
    qualifications: Mapping[str, Mapping[str, Any]],
    ordered_sample_ids: Sequence[str],
    schema_path: Path,
    git_commit: str | None,
) -> dict[str, Any]:
    """Build a preflight manifest without asserting an official dataset split."""
    ids = list(ordered_sample_ids)
    if not ids or len(ids) != len(set(ids)):
        raise PreflightError("ordered sample list must be non-empty and unique")
    adapted_by_id = {item["sample_id"]: item for item in adapted_samples}
    if len(adapted_by_id) != len(adapted_samples):
        raise PreflightError("duplicate adapted sample identity")
    unknown = sorted(set(ids) - set(adapted_by_id))
    if unknown:
        raise PreflightError(f"sample list is outside canonical dataset: {unknown}")
    missing_qualification = sorted(set(ids) - set(qualifications))
    if missing_qualification:
        raise PreflightError(f"missing qualification records: {missing_qualification}")

    entries = []
    for sample_id in ids:
        adapted = adapted_by_id[sample_id]
        qualification = qualifications[sample_id]
        entries.append({
            "sample_id": sample_id,
            "family_id": adapted["family_id"],
            "canonical_input_sha256": qualification.get("input_sample_sha256"),
            "qualification_status": qualification.get("status"),
            "qualification_record_sha256": hash_value(qualification),
            "officially_eligible": qualification.get("status") == "READY",
        })
    base = {
        "manifest_version": PREFLIGHT_VERSION,
        "purpose": "OFFLINE_PREFLIGHT_ONLY",
        "official_dataset": False,
        "official_split": False,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256(dataset_path),
        "schema_path": str(schema_path.resolve()),
        "schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
        "ordered_sample_ids": ids,
        "sample_list_sha256": hash_value(ids),
        "samples": entries,
        "git_commit": git_commit,
    }
    return {**base, "manifest_sha256": hash_value(base)}


def _code_hash() -> str:
    root = Path(__file__).resolve().parents[1]
    return hash_files([
        root / "pipeline" / "canonical_adapter.py",
        root / "pipeline" / "context_policy.py",
        root / "pipeline" / "offline_integration.py",
        root / "pipeline" / "preflight.py",
        root / "pipeline" / "run_state.py",
    ])


def _result_map_in_order(results: Sequence[Mapping[str, Any]], ids: Sequence[str]) -> list[dict[str, Any]]:
    by_id = {item.get("sample_id"): dict(item) for item in results}
    if len(by_id) != len(results):
        raise PreflightError("duplicate result identity")
    return [by_id[item] for item in ids if item in by_id]


def run_offline_preflight(
    *,
    dataset_path: Path,
    adapted_samples: Sequence[Mapping[str, Any]],
    qualifications: Mapping[str, Mapping[str, Any]],
    ordered_sample_ids: Sequence[str],
    schema_path: Path,
    output_root: Path,
    run_prefix: str,
    client_factory: Callable[[str], StructuredLLMClient],
    context_policy: ContextPolicy | None = None,
    git_commit: str | None = None,
    allow_needs_approval_for_preflight: bool = True,
    max_new_samples_per_pipeline: int | None = None,
) -> dict[str, Any]:
    """Run/resume an injected-mock preflight with one immutable list for P0/P1/P2.

    NEEDS_APPROVAL is permitted only for this explicitly non-official preflight.
    BLOCKED, OUT_OF_SCOPE and NOT_ASSESSED always fail closed.
    """
    policy = context_policy or ContextPolicy()
    manifest = build_dataset_manifest(
        dataset_path=dataset_path,
        adapted_samples=adapted_samples,
        qualifications=qualifications,
        ordered_sample_ids=ordered_sample_ids,
        schema_path=schema_path,
        git_commit=git_commit,
    )
    disallowed = []
    for entry in manifest["samples"]:
        status = entry["qualification_status"]
        if status == "READY":
            continue
        if status == "NEEDS_APPROVAL" and allow_needs_approval_for_preflight:
            continue
        disallowed.append(f"{entry['sample_id']}:{status}")
    if disallowed:
        raise PreflightError(f"qualification gate rejected samples: {disallowed}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "dataset_manifest.json"
    if manifest_path.exists():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored != manifest:
            raise PreflightError("PREFLIGHT_MANIFEST_MISMATCH")
    else:
        _atomic_create_json(manifest_path, manifest)

    adapted_by_id = {item["sample_id"]: item for item in adapted_samples}
    ids = list(ordered_sample_ids)
    results_by_pipeline: dict[str, list[dict[str, Any]]] = {}
    run_specs: dict[str, dict[str, Any]] = {}
    complete = True
    for pipeline in PIPELINES:
        lower = pipeline.lower()
        spec = build_run_spec(
            run_id=f"{run_prefix}-{lower}",
            dataset_sha256=manifest["dataset_sha256"],
            sample_ids=ids,
            pipeline_id=pipeline,
            pipeline_version=PREFLIGHT_VERSION,
            model_id="offline-mock",
            model_version="deterministic-1",
            parameters={"temperature": 0.0, "network_enabled": False},
            retry_policy={"max_attempts": 1, "retry_delays_sec": []},
            context_policy_sha256=policy.sha256(),
            prompt_hashes=prompt_hashes(lower),
            code_sha256=_code_hash(),
            git_commit=git_commit,
        )
        workspace = RunWorkspace(output_root / "runs", spec)
        workspace.initialize()
        existing = workspace.load_results()
        completed_ids = {item["sample_id"] for item in existing}
        new_count = 0
        client = client_factory(pipeline)
        for sample_id in ids:
            if sample_id in completed_ids:
                continue
            if max_new_samples_per_pipeline is not None and new_count >= max_new_samples_per_pipeline:
                break
            result = run_canonical_sample(lower, adapted_by_id[sample_id], client, policy)
            workspace.write_result(result)
            workspace.append_event({
                "event": "SAMPLE_RESULT_WRITTEN",
                "sample_id": sample_id,
                "execution_status": result["execution_status"],
            })
            new_count += 1
        ordered_results = _result_map_in_order(workspace.load_results(), ids)
        if len(ordered_results) != len(ids):
            complete = False
        results_by_pipeline[pipeline] = ordered_results
        run_specs[pipeline] = spec

    evaluation = evaluate_positive_only_ablation(results_by_pipeline, ids) if complete else None
    return {
        "preflight_version": PREFLIGHT_VERSION,
        "purpose": "OFFLINE_PREFLIGHT_ONLY",
        "official_pilot": False,
        "complete": complete,
        "ordered_sample_ids": ids,
        "sample_list_sha256": manifest["sample_list_sha256"],
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "run_spec_sha256_by_pipeline": {
            pipeline: spec["spec_sha256"] for pipeline, spec in run_specs.items()
        },
        "result_count_by_pipeline": {
            pipeline: len(results) for pipeline, results in results_by_pipeline.items()
        },
        "evaluation": evaluation,
    }
