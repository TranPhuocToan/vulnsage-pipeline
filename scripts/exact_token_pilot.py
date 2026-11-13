"""Exact, local-only Qwen tokenizer preflight for a canonical pilot dataset.

This module never creates an LLM client.  It records static prompt counts and
keeps P1/P2 synthesis explicitly runtime-dependent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from pipeline.canonical_adapter import load_and_adapt_dataset
from pipeline.context_diagnostics import candidate_policy
from pipeline.context_policy import ContextPolicy
from pipeline.offline_integration import render_static_prompt_plan


TOKENIZER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TOKENIZER_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
TOKENIZER_ASSET_SHA256 = {
    "config.json": "c0242402ad6a13b331ea320feea8c7e3776ffb7a4eff0757b9cd667e116d9a28",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "959e7f1d9a1b7641a6d6ce05ca97b75c7894fcb66cbe5a040406458fb1128ee4",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
PINNED_MODEL_DIGEST = "dae161e27b0e90dd1856c8bb3209201fd6736d8eb66298e75ed87571486f4364"
OBSERVED_SERVER_OVERHEAD_TOKENS = 29


class ExactTokenPreflightError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tokenizer_assets(snapshot: Path) -> list[dict[str, Any]]:
    results = []
    for name, expected in TOKENIZER_ASSET_SHA256.items():
        path = snapshot / name
        actual = sha256_file(path) if path.is_file() else None
        results.append({
            "name": name,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "MATCH" if actual == expected else ("MISSING" if actual is None else "MISMATCH"),
        })
    if any(item["status"] != "MATCH" for item in results):
        raise ExactTokenPreflightError("TOKENIZER_ASSET_INTEGRITY_FAILED")
    return results


def classify_count(count: int, raw_limit: int, overhead: int) -> str:
    if type(count) is not int or count < 0:
        raise ExactTokenPreflightError("TOKEN_COUNT_INVALID")
    return "FIT_WITH_OBSERVED_OVERHEAD" if count + overhead <= raw_limit else "OVERFLOW"


def build_exact_report(
    *,
    dataset_path: Path,
    counter: Callable[[str], int],
    tokenizer_metadata: Mapping[str, Any],
    asset_results: list[dict[str, Any]],
    model_digest: str,
    num_ctx: int = 32768,
    num_predict: int = 2048,
) -> dict[str, Any]:
    if model_digest != PINNED_MODEL_DIGEST:
        raise ExactTokenPreflightError("MODEL_DIGEST_NOT_EMPIRICALLY_CORRELATED")
    raw_limit = num_ctx - num_predict
    policies = (("HISTORICAL_DEFAULT", ContextPolicy()), ("CANDIDATE_NEEDS_APPROVAL", candidate_policy()))
    samples = load_and_adapt_dataset(dataset_path)
    policy_reports = []
    for policy_status, policy in policies:
        sample_reports = []
        static_fit_samples = 0
        blocked_samples = 0
        for sample in samples:
            pipeline_reports: dict[str, Any] = {}
            sample_static_fit = True
            preparation_error = None
            for pipeline in ("p0", "p1", "p2"):
                try:
                    plan = render_static_prompt_plan(pipeline, sample, policy)
                except Exception as exc:
                    preparation_error = f"{type(exc).__name__}:{exc}"
                    sample_static_fit = False
                    pipeline_reports[pipeline.upper()] = {
                        "static_calls": [],
                        "static_fit": False,
                        "status": "BLOCKED_CONTEXT_PREPARATION",
                        "error": preparation_error,
                        "runtime_dependent_stages": [],
                    }
                    continue
                calls = []
                for prompt in plan["static_prompts"]:
                    count = counter(prompt["prompt"])
                    status = classify_count(count, raw_limit, OBSERVED_SERVER_OVERHEAD_TOKENS)
                    sample_static_fit = sample_static_fit and status != "OVERFLOW"
                    calls.append({
                        "stage": prompt["stage"],
                        "prompt_sha256": prompt["prompt_sha256"],
                        "utf8_bytes": prompt["utf8_bytes"],
                        "raw_tokenizer_tokens": count,
                        "observed_server_overhead_tokens": OBSERVED_SERVER_OVERHEAD_TOKENS,
                        "effective_input_tokens": count + OBSERVED_SERVER_OVERHEAD_TOKENS,
                        "input_limit": raw_limit,
                        "remaining_margin": raw_limit - count - OBSERVED_SERVER_OVERHEAD_TOKENS,
                        "status": status,
                    })
                pipeline_reports[pipeline.upper()] = {
                    "static_calls": calls,
                    "static_fit": all(item["status"] != "OVERFLOW" for item in calls),
                    "runtime_dependent_stages": [
                        {"stage": stage, "status": "RUNTIME_EXACT_GUARD_REQUIRED"}
                        for stage in plan["runtime_dependent_stages"]
                    ],
                }
            if sample_static_fit:
                static_fit_samples += 1
            if preparation_error:
                blocked_samples += 1
            sample_reports.append({
                "sample_id": sample["sample_id"],
                "source_id": sample["audit_only"]["source"]["source_row_id"],
                "all_static_prompts_fit": sample_static_fit,
                "preparation_error": preparation_error,
                "pipelines": pipeline_reports,
            })
        policy_reports.append({
            "policy_status": policy_status,
            "policy_version": policy.policy_version,
            "policy_sha256": policy.sha256(),
            "code_representation": policy.code_representation,
            "static_fit_samples": static_fit_samples,
            "blocked_context_preparation_samples": blocked_samples,
            "static_overflow_or_blocked_samples": len(samples) - static_fit_samples,
            "samples": sample_reports,
        })
    return {
        "artifact_type": "pilot20_exact_static_token_preflight",
        "artifact_version": "1.0",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "sample_count": len(samples),
        "experimental_llm_calls": 0,
        "model": {
            "ollama_id": "qwen2.5-coder:7b",
            "digest": model_digest,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "raw_input_limit": raw_limit,
        },
        "tokenizer": dict(tokenizer_metadata),
        "tokenizer_assets": asset_results,
        "tokenizer_runtime_correspondence": {
            "status": "EMPIRICALLY_VERIFIED_FOR_PINNED_MODEL_DIGEST_ONLY",
            "basis": "Historical engineering-smoke observations: Ollama prompt_eval_count minus raw local tokenizer count was 29 on every recorded call.",
            "observed_overhead_tokens": OBSERVED_SERVER_OVERHEAD_TOKENS,
            "limitation": "The causal source of the 29-token overhead was not independently identified; future runtime must match the pinned model digest and re-check the invariant.",
        },
        "runtime_synthesis": "P1/P2 synthesis depends on preceding model output and remains protected by a pre-call exact guard; static preflight cannot prove it.",
        "policies": policy_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-digest", default=PINNED_MODEL_DIGEST)
    args = parser.parse_args()

    assets = verify_tokenizer_assets(args.tokenizer_snapshot)
    import tokenizers
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    metadata = {
        "id": TOKENIZER_ID,
        "revision": TOKENIZER_REVISION,
        "class": type(tokenizer).__name__,
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "add_special_tokens": False,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "model_max_length": tokenizer.model_max_length,
    }
    report = build_exact_report(
        dataset_path=args.dataset,
        counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        tokenizer_metadata=metadata,
        asset_results=assets,
        model_digest=args.model_digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ExactTokenPreflightError("OUTPUT_EXISTS_REFUSING_OVERWRITE")
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
