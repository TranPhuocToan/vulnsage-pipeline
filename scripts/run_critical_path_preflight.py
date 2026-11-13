"""One-command, no-network P0/P1/P2 development preflight.

The command exercises canonical adaptation, one immutable ordered list, mock
P0/P1/P2 execution, checkpoint/resume, and positive-only evaluation.  It never
constructs Ollama or another network transport.  A main-stage invocation is
fail-closed unless an attributable pilot PASS gate is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.canonical_adapter import load_and_adapt_dataset
from pipeline.llm_client import StructuredLLMClient
from pipeline.preflight import PreflightError, run_offline_preflight
from pipeline.run_state import hash_value


MODEL_CONTRACT = {
    "model_id": "qwen2.5-coder:7b",
    "num_ctx": 32768,
    "num_predict": 2048,
    "temperature": 0.0,
    "seed": 25550029,
    "stream": False,
    "retry_policy": {"max_attempts": 3, "retry_delays_sec": [1.0, 2.0, 4.0]},
    "exact_token_policy": "STATIC_PREFLIGHT_PLUS_PRE_CALL_RUNTIME_GUARD",
}


class CriticalPathPreflightError(ValueError):
    pass


class DeterministicMockTransport:
    def __call__(self, prompt: str) -> tuple[str, dict[str, int]]:
        if '"evidence_integration"' in prompt or '"global_synthesis"' in prompt:
            payload = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "offline mock"}
        elif '"dependency_analysis"' in prompt and '"num_cross_file_dependencies"' in prompt:
            payload = {
                "dependency_analysis": "offline mock", "dependencies": [],
                "num_cross_file_dependencies": 0, "overall_coupling": "LOOSE",
            }
        elif '"function_summary"' in prompt and '"local_verdict"' in prompt:
            payload = {
                "file_name": "offline-fixture.c", "function_summary": "offline mock",
                "changes_summary": "offline mock", "security_purpose": "offline mock",
                "local_verdict": "CONTRIBUTES_TO_FIX", "confidence": "HIGH",
                "key_observations": ["offline mock"],
            }
        else:
            payload = {"verdict": "PATCHED", "confidence": "HIGH", "reasoning": "offline mock"}
        return json.dumps(payload), {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


def load_qualifications(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records") if isinstance(value, Mapping) else None
    if not isinstance(records, list):
        raise CriticalPathPreflightError("qualification JSON must contain records[]")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("sample_id"):
            continue
        sample_id = str(record["sample_id"])
        if sample_id in result:
            raise CriticalPathPreflightError(f"duplicate qualification sample_id: {sample_id}")
        result[sample_id] = record
    return result


def validate_main_gate(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise CriticalPathPreflightError("MAIN_LOCKED_PILOT_PASS_GATE_REQUIRED")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS",
        "evaluation_scope": "POSITIVE_ONLY_KNOWN_FIX_COMMIT_POST_STATES",
        "model_id": MODEL_CONTRACT["model_id"],
    }
    if not isinstance(value, dict) or any(value.get(key) != expected for key, expected in required.items()):
        raise CriticalPathPreflightError("MAIN_LOCKED_INVALID_PILOT_PASS_GATE")
    if not all(isinstance(value.get(key), str) and len(value[key]) == 64 for key in ("dataset_sha256", "sample_list_sha256")):
        raise CriticalPathPreflightError("MAIN_LOCKED_PILOT_HASH_PROVENANCE_REQUIRED")
    return value


def ensure_contract(path: Path, contract: Mapping[str, Any]) -> None:
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != contract:
            raise CriticalPathPreflightError("OFFLINE_CONTRACT_MISMATCH")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise CriticalPathPreflightError("OFFLINE_CONTRACT_ALREADY_EXISTS")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_summary_artifact(
    summary: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind offline evaluation output to its inputs without implying research results."""
    artifact = {
        "artifact_type": "critical_path_offline_preflight_summary",
        "artifact_version": "november-fast-track-preflight-summary-1.0",
        "purpose": "OFFLINE_PREFLIGHT_ONLY",
        "official_pilot": False,
        "complete": bool(summary["complete"]),
        "execution_transport": "DETERMINISTIC_OFFLINE_MOCK",
        "experimental_llm_calls": 0,
        "contract_sha256": contract["contract_sha256"],
        "ordered_sample_ids": list(summary["ordered_sample_ids"]),
        "sample_list_sha256": summary["sample_list_sha256"],
        "dataset_manifest_sha256": summary["dataset_manifest_sha256"],
        "run_spec_sha256_by_pipeline": dict(summary["run_spec_sha256_by_pipeline"]),
        "result_count_by_pipeline": dict(summary["result_count_by_pipeline"]),
        "evaluation": summary["evaluation"],
        "evaluation_provenance": {
            "evaluator": "pipeline.evaluator.evaluate_positive_only_ablation",
            "scope": "POSITIVE_ONLY_OFFLINE_MOCK_SCHEMA_AND_RESUME_TEST",
            "ground_truth_review": "PENDING_NOT_PROMOTED",
            "research_metric_claim_authorized": False,
        },
    }
    return {**artifact, "artifact_sha256": hash_value(artifact)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--qualifications", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--run-prefix", default="november-development-dry-run")
    parser.add_argument("--stage", choices=("development-pilot", "main"), default="development-pilot")
    parser.add_argument("--pilot-pass-gate", type=Path)
    parser.add_argument("--max-new-samples-per-pipeline", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "main":
        validate_main_gate(args.pilot_pass_gate)
    if len(args.sample_id) != len(set(args.sample_id)):
        raise CriticalPathPreflightError("duplicate ordered sample ID")
    if len(args.git_commit) != 40:
        raise CriticalPathPreflightError("full experiment Git SHA required")
    qualifications = load_qualifications(args.qualifications)
    adapted = load_and_adapt_dataset(args.dataset, schema_path=args.schema)
    contract_base = {
        "purpose": "OFFLINE_PREFLIGHT_ONLY",
        "official_pilot": False,
        "stage": args.stage,
        "ordered_sample_ids": list(args.sample_id),
        "planned_experiment_contract": MODEL_CONTRACT,
        "experimental_llm_calls": 0,
        "main_locked_without_pilot_pass": True,
        "git_commit": args.git_commit,
    }
    contract = {**contract_base, "contract_sha256": hash_value(contract_base)}
    ensure_contract(args.output_root / "offline_preflight_contract.json", contract)

    def client_factory(_pipeline: str) -> StructuredLLMClient:
        return StructuredLLMClient(
            DeterministicMockTransport(), max_attempts=1, retry_delays=[0.0]
        )

    summary = run_offline_preflight(
        dataset_path=args.dataset,
        adapted_samples=adapted,
        qualifications=qualifications,
        ordered_sample_ids=args.sample_id,
        schema_path=args.schema,
        output_root=args.output_root,
        run_prefix=args.run_prefix,
        client_factory=client_factory,
        git_commit=args.git_commit,
        allow_needs_approval_for_preflight=True,
        max_new_samples_per_pipeline=args.max_new_samples_per_pipeline,
    )
    summary_artifact = build_summary_artifact(summary, contract)
    if summary["complete"]:
        summary_path = args.output_root / "offline_preflight_summary.json"
    else:
        count_hash = hash_value(summary["result_count_by_pipeline"])[:12]
        summary_path = args.output_root / f"offline_preflight_checkpoint_{count_hash}.json"
    ensure_contract(summary_path, summary_artifact)
    print(json.dumps({
        "complete": summary["complete"],
        "purpose": summary["purpose"],
        "result_count_by_pipeline": summary["result_count_by_pipeline"],
        "sample_list_sha256": summary["sample_list_sha256"],
        "experimental_llm_calls": 0,
        "summary_path": str(summary_path.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CriticalPathPreflightError, PreflightError) as exc:
        raise SystemExit(f"BLOCKED:{exc}") from exc
