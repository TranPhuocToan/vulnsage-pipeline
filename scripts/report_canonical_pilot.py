"""Create honest offline readiness artifacts from a canonical pilot prototype."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_validation.qualification import qualify_sample
from pipeline.canonical_adapter import load_and_adapt_dataset
from pipeline.context_policy import ContextPolicy
from pipeline.offline_integration import render_static_prompt_plan


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")
        return
    path.write_bytes(data)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def build_reports(dataset_path: Path, run_path: Path, schema_path: Path, output_dir: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    adapted = load_and_adapt_dataset(dataset_path, schema_path=schema_path)
    adapted_by_id = {item["sample_id"]: item for item in adapted}
    run_by_source = {str(item["source_id"]): item for item in run["records"]}
    policy = ContextPolicy()
    rows: list[dict[str, Any]] = []
    token_samples: list[dict[str, Any]] = []
    for sample in dataset["samples"]:
        source_id = str(sample["source"]["source_row_id"])
        qualification = qualify_sample(sample)
        record = run_by_source[source_id]
        static = []
        for pipeline in ("p0", "p1", "p2"):
            plan = render_static_prompt_plan(pipeline, adapted_by_id[sample["sample_id"]], policy)
            static.extend(
                {
                    "pipeline": pipeline.upper(),
                    "stage": item["stage"],
                    "prompt_sha256": item["prompt_sha256"],
                    "utf8_bytes": item["utf8_bytes"],
                    "exact_input_tokens": None,
                    "status": "BLOCKED_EXACT_QWEN_TOKENIZER_UNAVAILABLE",
                }
                for item in plan["static_prompts"]
            )
        blockers = list(qualification["reason_codes"])
        blockers += ["SECURITY_REVIEW_HUMAN_DECISION_REQUIRED", "GROUND_TRUTH_REVIEW_PENDING"]
        blockers += ["EXACT_QWEN_TOKEN_PREFLIGHT_NOT_RUN"]
        rows.append(
            {
                "source_id": source_id,
                "sample_id": sample["sample_id"],
                "cve_id": sample["identity"]["cve_id"],
                "project": sample["identity"]["project"],
                "repository": record["repository"],
                "fix_commit": sample["identity"]["fix_commit_hash"],
                "parent_commit": sample["identity"]["parent_commit_hash"],
                "changed_files": len(sample["files"]),
                "canonical_status": "TECHNICALLY_PREPARED",
                "commit_verification": sample["commit_verification"]["status"],
                "quality_status": sample["quality"]["validation_status"],
                "qualification_status": qualification["status"],
                "security_review": "HUMAN_REVIEW_REQUIRED",
                "ground_truth_review": sample["ground_truth"]["review_status"],
                "p0_token_status": "BLOCKED",
                "p1_token_status": "BLOCKED",
                "p2_token_status": "BLOCKED",
                "context_policy_sha256": policy.sha256(),
                "eligible": False,
                "remaining_blockers": "|".join(sorted(set(blockers))),
                "evidence_snapshot_sha256": record["evidence_snapshot_sha256"],
                "retrieval_manifest_sha256": record["retrieval_manifest_sha256"],
            }
        )
        token_samples.append(
            {
                "source_id": source_id,
                "sample_id": sample["sample_id"],
                "context_policy_sha256": policy.sha256(),
                "model_id": "qwen2.5-coder:7b",
                "num_ctx": 32768,
                "reserved_output_tokens": 2048,
                "available_input_tokens": 30720,
                "tokenizer_status": "BLOCKED_UNAVAILABLE",
                "p0_status": "BLOCKED",
                "p1_status": "BLOCKED",
                "p2_status": "BLOCKED",
                "eligible": False,
                "static_prompts": static,
                "runtime_dependent_stages": ["P1:global_synthesis", "P2:evidence_synthesis"],
                "reason": "No exact Qwen tokenizer is installed; heuristic counts are not reported as exact.",
            }
        )

    matrix_path = output_dir / "canonicalization_matrix.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_once(matrix_path, stream.getvalue())

    qualification_counts = Counter(row["qualification_status"] for row in rows)
    summary = {
        "artifact_type": "canonical_pilot20_readiness_summary",
        "source_only_candidates": run["source_only_candidates"],
        "canonicalized_samples": len(rows),
        "schema_valid_samples": len(adapted),
        "context_fit_exact": 0,
        "human_review_required": len(rows),
        "qualified_ready": qualification_counts.get("READY", 0),
        "qualification_status_counts": dict(sorted(qualification_counts.items())),
        "ground_truth_pending": sum(row["ground_truth_review"] == "PENDING" for row in rows),
        "security_semantics_verified": 0,
        "dataset_sha256": _sha(dataset_path),
        "source_csv_sha256": dataset["source_csv_sha256"],
        "context_policy_sha256": policy.sha256(),
        "experimental_llm_calls": 0,
        "official_pilot": False,
        "inference_authorized": False,
    }
    _write_once(output_dir / "canonicalization_summary.json", _json_text(summary))
    token_report = {
        "artifact_type": "exact_token_preflight",
        "status": "BLOCKED_EXACT_TOKENIZER_UNAVAILABLE",
        "exact": False,
        "heuristic_substituted": False,
        "model_id": "qwen2.5-coder:7b",
        "parameters": {"num_ctx": 32768, "num_predict": 2048, "temperature": 0, "seed": 25550029},
        "context_policy_sha256": policy.sha256(),
        "samples": token_samples,
        "experimental_llm_calls": 0,
    }
    _write_once(output_dir / "exact_token_preflight.json", _json_text(token_report))

    header = (
        "# Pilot review packet\n\n"
        "20 mẫu dưới đây đã có canonical commit evidence và schema validation. "
        "Đây không phải security approval hoặc official pilot. Tất cả ground truth vẫn PENDING.\n\n"
        "| source | sample_id | CVE | upstream | commit | canonical | security_review | ground_truth | P0 | P1 | P2 | remaining_blocker |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = "".join(
        f"| {row['source_id']} | `{row['sample_id']}` | {row['cve_id']} | {row['repository']} | "
        f"`{row['fix_commit']}` | {row['canonical_status']} | {row['security_review']} | "
        f"{row['ground_truth_review']} | {row['p0_token_status']} | {row['p1_token_status']} | "
        f"{row['p2_token_status']} | {row['remaining_blockers']} |\n"
        for row in rows
    )
    footer = (
        "\n## Researcher checklist\n\n"
        "- Xác minh CVE ↔ exact commit bằng nguồn upstream/advisory độc lập.\n"
        "- Đánh giá security relevance của từng file và tính đầy đủ của commit unit.\n"
        "- Quyết định derived-only waiver riêng cho từng reason code; không bulk approve.\n"
        "- Giữ review_status=PENDING cho tới khi quyết định có reviewer provenance.\n"
        "- Chạy exact Qwen tokenizer preflight trên canonical prompts trước inference.\n"
        "- Không dùng mẫu OUT_OF_SCOPE/BLOCKED trong official pilot khi chưa xử lý.\n"
    )
    _write_once(output_dir / "pilot_review_packet.md", header + body + footer)

    decision = f"""# Pilot readiness decision

## Decision

**NO-GO cho pilot inference tại thời điểm hiện tại.**

## Evidence

- Source-only candidates: {summary['source_only_candidates']}.
- Canonicalized và schema-valid: {summary['canonicalized_samples']}.
- Qualification: {dict(sorted(qualification_counts.items()))}.
- Exact context-fit: 0/20 vì exact Qwen tokenizer chưa khả dụng.
- Human security review required: 20/20.
- Ground truth PENDING: {summary['ground_truth_pending']}/20.
- Qualified READY: {summary['qualified_ready']}.
- Experimental LLM/API calls: 0.

Canonicalization chứng minh commit/parent/patch/blob provenance, không chứng minh security correctness.
Không tạo official manifest trước khi có exact-token PASS, attributable review decisions và qualification READY.

## Deterministic next gate

1. Review trước hai mẫu commit-fidelity sạch `18922` và `18867`.
2. Review derived gaps của 18 mẫu còn lại; `5530` đang OUT_OF_SCOPE và `10791` đang BLOCKED.
3. Cài/khóa exact Qwen tokenizer trong môi trường preflight được phê duyệt và đếm static prompts; P1/P2 synthesis vẫn giữ runtime guard.
4. Tạo review decisions có provenance, chạy lại qualification và chỉ package các mẫu READY.
5. Nếu dưới 20 READY, dùng deterministic replacement từ 121 candidates; không hạ gate.

Replacement pool source-only đã được khóa tại
`F:\\UIT\\KLTN\\code\\VulnSage\\data\\pilot\\source_only_candidates_40_20261111.json`.
Hai mươi ID bổ sung theo đúng thứ tự deterministic là:
`16561, 19063, 1094, 16622, 10002, 18392, 16611, 7241, 17570, 16889, 2148, 18981, 17227, 13018, 18757, 16859, 18782, 2424, 17013, 18523`.
Đây chỉ là replacement pool source-only; chưa canonical, chưa review và chưa được phép inference.
"""
    _write_once(output_dir / "pilot_readiness_decision.md", decision)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = build_reports(args.dataset, args.run, args.schema, args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
