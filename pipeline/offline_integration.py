"""Canonical-data execution path shared by offline P0/P1/P2 tests.

No transport is created here.  The caller must inject ``StructuredLLMClient``;
therefore importing or testing this module cannot call a paid API.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

from pipeline.context_policy import (
    ContextOverflowError,
    ContextPolicy,
    enforce_context_limit,
    prepare_context,
)
from pipeline.llm_client import StructuredCallResult, StructuredLLMClient


PROMPT_FILES = {
    "p0": ["p0_think_verify.txt"],
    "p1": ["p1_local_analysis.txt", "p1_global_synthesis.txt"],
    "p2": ["p1_local_analysis.txt", "p2_dependency_extract.txt", "p2_evidence_synthesis.txt"],
}


def _prompts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def _load_prompt(name: str) -> str:
    return (_prompts_dir() / name).read_text(encoding="utf-8")


def _replace(template: str, values: Mapping[str, Any]) -> str:
    result = template
    for key, value in values.items():
        result = result.replace("{" + key + "}", str(value))
    return result


def prompt_hashes(pipeline_id: str) -> Dict[str, str]:
    return {
        name: hashlib.sha256((_prompts_dir() / name).read_bytes()).hexdigest()
        for name in PROMPT_FILES[pipeline_id]
    }


def render_static_prompt_plan(
    pipeline_id: str,
    adapted: Mapping[str, Any],
    policy: ContextPolicy | None = None,
) -> Dict[str, Any]:
    """Render every prompt whose content is known before any model call.

    P1/P2 synthesis prompts depend on earlier model outputs and are explicitly
    marked runtime-dependent instead of being represented by fake content.
    """
    pipeline_id = pipeline_id.lower()
    if pipeline_id not in PROMPT_FILES:
        raise ValueError(f"unknown pipeline: {pipeline_id}")
    policy = policy or ContextPolicy()
    context = prepare_context(adapted, policy)
    metadata = context["metadata"]
    files = adapted["model_input"]["files"]
    prompts: List[Dict[str, Any]] = []
    runtime_dependent: List[str] = []
    if pipeline_id == "p0":
        code_block = "\n\n".join(item["code_block"] for item in context["files"])
        prompt = _replace(_load_prompt("p0_think_verify.txt"), {
            **metadata,
            "code_block": code_block,
            "git_diff": context["diff_bundle"]["text"],
        })
        prompts.append({"stage": "p0.verdict", "prompt": prompt})
    else:
        local_template = _load_prompt("p1_local_analysis.txt")
        for index, (file_data, file_context) in enumerate(zip(files, context["files"])):
            prompt = _replace(local_template, {
                **metadata,
                "file_name": file_data["display_path"],
                "file_index": index + 1,
                "total_files": len(files),
                "vulnerable_code": file_context["before_code"],
                "patched_code": file_context["after_code"],
                "git_diff": file_context["diff"],
            })
            prompts.append({"stage": f"{pipeline_id}.local.{index}", "prompt": prompt})
        if pipeline_id == "p2":
            file_list = "\n".join(
                f"- {item['change_type']}: {item['path_before']} -> {item['path_after']}"
                for item in files
            )
            prompt = _replace(_load_prompt("p2_dependency_extract.txt"), {
                **metadata,
                "num_files": len(files),
                "file_list": file_list,
                "full_git_diff": context["diff_bundle"]["text"],
            })
            prompts.append({"stage": "p2.dependency", "prompt": prompt})
        runtime_dependent.append(f"{pipeline_id}.synthesis")
    return {
        "pipeline": pipeline_id.upper(),
        "context_policy_sha256": policy.sha256(),
        "representation_provenance": [
            item["representation_provenance"] for item in context["files"]
            if "representation_provenance" in item
        ],
        "static_prompts": [
            {
                "stage": item["stage"],
                "prompt": item["prompt"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest(),
                "utf8_bytes": len(item["prompt"].encode("utf-8")),
            }
            for item in prompts
        ],
        "runtime_dependent_stages": runtime_dependent,
    }


def _attempt_summary(calls: List[StructuredCallResult]) -> Dict[str, Any]:
    attempts = [attempt for call in calls for attempt in call.attempts]
    fields = ("input_tokens", "output_tokens", "total_tokens")
    totals: Dict[str, Any] = {}
    complete = True
    for field in fields:
        values = [attempt[field] for attempt in attempts]
        if any(value is None for value in values):
            totals[field] = None
            complete = False
        else:
            totals[field] = sum(values)
    totals["usage_complete"] = complete
    return {
        "token_usage": totals,
        "attempts": attempts,
        "num_attempts": len(attempts),
        "total_attempt_duration_sec": round(
            sum(item["duration_sec"] for item in attempts), 9
        ),
    }


def _technical_result(
    pipeline_id: str,
    adapted: Mapping[str, Any],
    calls: List[StructuredCallResult],
    reason: str,
    context_hashes: List[str],
    pre_call_token_counts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "sample_id": adapted["sample_id"],
        "pipeline": pipeline_id.upper(),
        "execution_status": "TECHNICAL_ERROR",
        "prediction": None,
        "technical_error": reason,
        "context_hashes": context_hashes,
        "exact_token_preflight": "EXACT" if pre_call_token_counts else "NOT_RUN",
        "pre_call_token_counts": pre_call_token_counts,
        **_attempt_summary(calls),
    }


def _invoke(
    client: StructuredLLMClient,
    prompt: str,
    kind: str,
    stage: str,
    policy: ContextPolicy,
    token_counter: Callable[[str], int] | None,
    model_input_token_limit: int | None,
    pre_call_token_counts: List[Dict[str, Any]],
) -> StructuredCallResult:
    enforce_context_limit(prompt, stage, policy)
    if token_counter is not None:
        measured = token_counter(prompt)
        if type(measured) is not int or measured < 0:
            raise ValueError("token counter must return a non-negative int")
        pre_call_token_counts.append({"stage": stage, "input_tokens": measured})
        if model_input_token_limit is not None and measured > model_input_token_limit:
            raise ContextOverflowError(f"tokens:{stage}", measured, model_input_token_limit)
    return client.invoke(prompt, kind)


def run_canonical_sample(
    pipeline_id: str,
    adapted: Mapping[str, Any],
    client: StructuredLLMClient,
    policy: ContextPolicy | None = None,
    token_counter: Callable[[str], int] | None = None,
    model_input_token_limit: int | None = None,
) -> Dict[str, Any]:
    """Run one adapted sample while preserving the P0/P1/P2 method boundary."""
    pipeline_id = pipeline_id.lower()
    if pipeline_id not in PROMPT_FILES:
        raise ValueError(f"unknown pipeline: {pipeline_id}")
    policy = policy or ContextPolicy()
    calls: List[StructuredCallResult] = []
    context_hashes: List[str] = []
    pre_call_token_counts: List[Dict[str, Any]] = []
    try:
        context = prepare_context(adapted, policy)
        metadata = context["metadata"]
        files = adapted["model_input"]["files"]

        if pipeline_id == "p0":
            code_block = "\n\n".join(item["code_block"] for item in context["files"])
            prompt = _replace(_load_prompt("p0_think_verify.txt"), {
                **metadata,
                "code_block": code_block,
                "git_diff": context["diff_bundle"]["text"],
            })
            context_hashes.extend([item["sha256"] for item in context["files"]])
            context_hashes.append(context["diff_bundle"]["sha256"])
            call = _invoke(
                client, prompt, "verdict", "p0.verdict", policy,
                token_counter, model_input_token_limit, pre_call_token_counts,
            )
            calls.append(call)
            if call.status != "OK":
                return _technical_result("p0", adapted, calls, call.error_message or call.status, context_hashes, pre_call_token_counts)
            prediction = call.parsed["verdict"]

        else:
            local_results: List[Dict[str, Any]] = []
            local_template = _load_prompt("p1_local_analysis.txt")
            for index, (file_data, file_context) in enumerate(zip(files, context["files"])):
                prompt = _replace(local_template, {
                    **metadata,
                    "file_name": file_data["display_path"],
                    "file_index": index + 1,
                    "total_files": len(files),
                    "vulnerable_code": file_context["before_code"],
                    "patched_code": file_context["after_code"],
                    "git_diff": file_context["diff"],
                })
                context_hashes.append(file_context["sha256"])
                call = _invoke(
                    client, prompt, "local", f"{pipeline_id}.local.{index}", policy,
                    token_counter, model_input_token_limit, pre_call_token_counts,
                )
                calls.append(call)
                if call.status != "OK":
                    return _technical_result(pipeline_id, adapted, calls, call.error_message or call.status, context_hashes, pre_call_token_counts)
                local_results.append({
                    "file": file_data["display_path"],
                    "change_type": file_data["change_type"],
                    "analysis": call.parsed,
                })

            local_text = json.dumps(local_results, ensure_ascii=False, sort_keys=True)
            if pipeline_id == "p1":
                prompt = _replace(_load_prompt("p1_global_synthesis.txt"), {
                    **metadata,
                    "num_files": len(files),
                    "local_analyses_text": local_text,
                })
            else:
                file_list = "\n".join(
                    f"- {item['change_type']}: {item['path_before']} -> {item['path_after']}"
                    for item in files
                )
                dependency_prompt = _replace(_load_prompt("p2_dependency_extract.txt"), {
                    **metadata,
                    "num_files": len(files),
                    "file_list": file_list,
                    "full_git_diff": context["diff_bundle"]["text"],
                })
                context_hashes.append(context["diff_bundle"]["sha256"])
                dependency_call = _invoke(
                    client, dependency_prompt, "dependency", "p2.dependency", policy,
                    token_counter, model_input_token_limit, pre_call_token_counts,
                )
                calls.append(dependency_call)
                if dependency_call.status != "OK":
                    return _technical_result("p2", adapted, calls, dependency_call.error_message or dependency_call.status, context_hashes, pre_call_token_counts)
                prompt = _replace(_load_prompt("p2_evidence_synthesis.txt"), {
                    **metadata,
                    "num_files": len(files),
                    "local_analyses_text": local_text,
                    "dependency_evidence_table": json.dumps(
                        dependency_call.parsed, ensure_ascii=False, sort_keys=True
                    ),
                })
            final_call = _invoke(
                client, prompt, "verdict", f"{pipeline_id}.synthesis", policy,
                token_counter, model_input_token_limit, pre_call_token_counts,
            )
            calls.append(final_call)
            if final_call.status != "OK":
                return _technical_result(pipeline_id, adapted, calls, final_call.error_message or final_call.status, context_hashes, pre_call_token_counts)
            prediction = final_call.parsed["verdict"]

        return {
            "sample_id": adapted["sample_id"],
            "pipeline": pipeline_id.upper(),
            "execution_status": "COMPLETED",
            "prediction": prediction,
            "model_abstained": prediction == "NOT_SURE",
            "context_hashes": context_hashes,
            "context_policy_sha256": policy.sha256(),
            "representation_provenance": [
                item["representation_provenance"] for item in context["files"]
                if "representation_provenance" in item
            ],
            "prompt_hashes": prompt_hashes(pipeline_id),
            "exact_token_preflight": "EXACT" if token_counter is not None else "NOT_RUN",
            "pre_call_token_counts": pre_call_token_counts,
            **_attempt_summary(calls),
        }
    except ContextOverflowError as exc:
        return _technical_result(pipeline_id, adapted, calls, str(exc), context_hashes, pre_call_token_counts)
