"""Read-only prompt composition diagnostics; never constructs an LLM client."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from pipeline.canonical_adapter import load_and_adapt_dataset
from pipeline.context_policy import BEFORE_PLUS_DIFF, ContextPolicy, prepare_context
from pipeline.offline_integration import render_static_prompt_plan


def candidate_policy() -> ContextPolicy:
    return ContextPolicy(
        code_representation=BEFORE_PLUS_DIFF,
        policy_version="phase3d-lossless-delta-candidate-1.0",
    )


def compare_context_representations(
    adapted: Mapping[str, Any],
    token_counter: Callable[[str], int] | None = None,
    input_budget: int = 30_720,
) -> dict[str, Any]:
    """Compare complete prompt bytes and optional raw tokenizer counts.

    Component token counts are intentionally not summed: BPE tokenization
    across boundaries is not additive. No context is removed to meet a budget.
    """
    def measure(text: str) -> int | None:
        count = token_counter(text) if token_counter else None
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("invalid token counter")
        return count

    reports = []
    for policy in (ContextPolicy(), candidate_policy()):
        context = prepare_context(adapted, policy)
        files = adapted["model_input"]["files"]
        plans = []
        for pipeline in ("p0", "p1", "p2"):
            plan = render_static_prompt_plan(pipeline, adapted, policy)
            if pipeline == "p0":
                p0_prompt = plan["static_prompts"][0]["prompt"]
            calls = []
            for call in plan["static_prompts"]:
                count = measure(call["prompt"])
                calls.append({
                    "stage": call["stage"],
                    "prompt_sha256": call["prompt_sha256"],
                    "utf8_bytes": call["utf8_bytes"],
                    "raw_input_tokens": count,
                    "status": "NOT_MEASURED" if count is None else (
                        "FIT_RAW_PROMPT" if count <= input_budget else "CONTEXT_OVERFLOW"
                    ),
                })
            plans.append({
                "pipeline": pipeline.upper(),
                "calls": calls,
                "runtime_dependent_stages": plan["runtime_dependent_stages"],
            })
        p0_bytes = plans[0]["calls"][0]["utf8_bytes"]
        code_text = "\n\n".join(part["code_block"] for part in context["files"])
        rendered_code_bytes = len(code_text.encode("utf-8"))
        diff_bytes = len(context["diff_bundle"]["text"].encode("utf-8"))
        reports.append({
            "code_representation": policy.code_representation,
            "policy_version": policy.policy_version,
            "policy_sha256": policy.sha256(),
            "decision_status": "HUMAN_DECISION" if policy.code_representation == BEFORE_PLUS_DIFF else "HISTORICAL_DEFAULT",
            "p0_composition_utf8_bytes": {
                "rendered_code_blocks": rendered_code_bytes,
                "original_diff_bundle": diff_bytes,
                "instruction_template_and_metadata": p0_bytes - rendered_code_bytes - diff_bytes,
                "total": p0_bytes,
            },
            "p0_component_raw_tokens_nonadditive": {
                "rendered_code_blocks": measure(code_text),
                "original_diff_bundle": measure(context["diff_bundle"]["text"]),
                "instruction_template_and_metadata": measure(
                    p0_prompt.replace(code_text, "", 1).replace(context["diff_bundle"]["text"], "", 1)
                ),
            },
            "source_blob_utf8_bytes": [
                {
                    "path": f["display_path"],
                    "before": len((f["before_code"] or "").encode("utf-8")),
                    "after": len((f["after_code"] or "").encode("utf-8")),
                    "before_raw_tokens": measure(f["before_code"] or ""),
                    "after_raw_tokens": measure(f["after_code"] or ""),
                }
                for f in files
            ],
            "representation_provenance": [
                part["representation_provenance"] for part in context["files"]
                if "representation_provenance" in part
            ],
            "plans": plans,
        })
    return {
        "diagnostic_version": "phase3d-context-comparison-1.0",
        "sample_id": adapted["sample_id"],
        "exact_token_preflight": "NOT_RUN" if token_counter is None else "RAW_PROMPT_TOKENIZER_ONLY",
        "input_budget": input_budget,
        "server_template_overhead": "UNKNOWN",
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id", default="vs:e51ac3a94239:19047:canonical")
    parser.add_argument("--input-budget", type=int, default=30_720)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-revision")
    args = parser.parse_args()
    if args.input_budget <= 0:
        parser.error("input budget must be positive")
    tokenizer = None
    metadata = None
    if args.tokenizer_id:
        if not args.tokenizer_revision:
            parser.error("an exact tokenizer revision is required")
        import transformers
        import tokenizers
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer_id,
            revision=args.tokenizer_revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        metadata = {
            "id": args.tokenizer_id, "revision": args.tokenizer_revision,
            "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
            "class": type(tokenizer).__name__, "add_special_tokens": False,
        }
    samples = load_and_adapt_dataset(args.dataset)
    matches = [sample for sample in samples if sample["sample_id"] == args.sample_id]
    if len(matches) != 1:
        parser.error("exactly one matching sample is required")
    counter = None if tokenizer is None else lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    report = compare_context_representations(matches[0], counter, args.input_budget)
    report["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    report["tokenizer"] = metadata
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
