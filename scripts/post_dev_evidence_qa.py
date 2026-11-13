"""Create post-development QA artifacts without invoking a model or mutating results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.dependency_anchor_qa import validate_dependency_anchors


PRIORITY_IDS = ("18922", "18867", "13398", "16918", "18607", "8848")
EXPOSED_IDS = {"18922", "18867"}
PATCH_SUMMARIES = {
    "18922": "Ngăn double-free ở USB MIDI: caller chỉ free khi buffer còn tồn tại; helper cảnh báo và xóa con trỏ buffer sau free.",
    "18867": "Bỏ fast path Salsa20 gọi blkcipher_walk_done() khi zero-byte; áp dụng đồng nhất cho generic và x86.",
    "13398": "Biến allocation Metadata thành thao tác có thể thất bại và chỉ memcpy/gán type khi cấp phát thành công.",
    "16918": "Dùng guest_handle_subrange_okay() cho phần mảng còn lại và chặn overflow trong phép nhân count×size.",
    "18607": "Phân biệt /dev/mem page bị cấm với low-memory page cần zero-fill để tránh lộ System RAM.",
    "8848": "Kiểm tra JS object thật sự là nsXBLJSClass trước khi downcast/reuse; thêm assertion bảo vệ invariant.",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


def _read_json_from_zip(archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
    value = json.loads(archive.read(name).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument("--canonical-run", type=Path, required=True)
    parser.add_argument("--canonical-dataset", type=Path, required=True)
    parser.add_argument("--exact-result", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    canonical_run = json.loads(args.canonical_run.read_text(encoding="utf-8"))
    exact = json.loads(args.exact_result.read_text(encoding="utf-8"))
    records = {item["source_id"]: item for item in canonical_run["records"]}
    historical = next(item for item in exact["policies"] if item["policy_status"] == "HISTORICAL_DEFAULT")
    exact_by_id = {item["source_id"]: item for item in historical["samples"]}
    dataset_hash = sha256(args.canonical_dataset.read_bytes())
    if dataset_hash != exact["dataset_sha256"]:
        raise ValueError("CANONICAL_DATASET_HASH_MISMATCH")

    with zipfile.ZipFile(args.results_zip) as archive:
        attempts_raw = archive.read("vulnsage_dev_results/llm_attempts.jsonl")
        events = [json.loads(line) for line in attempts_raw.decode("utf-8").splitlines() if line.strip()]
        received = [event for event in events if event.get("event") == "received"]
        response_hash_matches = sum(
            sha256(event["raw_response"].encode("utf-8")) == event["response_sha256"]
            for event in received
        )
        result_objects = []
        for pipeline in ("P0", "P1", "P2"):
            for source_id in ("18922", "18867"):
                result_objects.append(_read_json_from_zip(
                    archive, f"vulnsage_dev_results/results/{pipeline}_{source_id}.json"
                ))
        calls = sum(len(item["result"]["attempts"]) for item in result_objects)
        input_tokens = sum(item["result"]["token_usage"]["input_tokens"] for item in result_objects)
        output_tokens = sum(item["result"]["token_usage"]["output_tokens"] for item in result_objects)
        all_completed = all(item["result"]["execution_status"] == "COMPLETED" for item in result_objects)
        retry_or_error = any(
            attempt.get("attempt") != 1 or attempt.get("status") != "OK"
            for item in result_objects for attempt in item["result"]["attempts"]
        )

        dependency_responses: dict[str, dict[str, Any]] = {}
        p0_responses: dict[str, dict[str, Any]] = {}
        for event in received:
            payload = json.loads(event["raw_response"])
            job = event["job"]
            source_id = job.split(":", 1)[1]
            if job.startswith("P2:") and isinstance(payload.get("dependencies"), list):
                dependency_responses[source_id] = payload
            if job.startswith("P0:") and "think" in payload and "verify" in payload:
                p0_responses[source_id] = payload

    anchor_reports = {}
    for source_id, dependency_evidence in dependency_responses.items():
        retrieval = args.evidence_root / source_id / "retrieval"
        manifest = json.loads((retrieval / "retrieval_manifest.json").read_text(encoding="utf-8"))
        diff_resource = next(item for item in manifest["resources"] if item["kind"] == "ORIGINAL_DIFF")
        anchor_reports[source_id] = validate_dependency_anchors(
            original_diff_bytes=(retrieval / diff_resource["local_path"]).read_bytes(),
            expected_diff_sha256=diff_resource["sha256"],
            dependency_evidence=dependency_evidence,
        )

    p0_quality = {}
    for source_id, payload in p0_responses.items():
        think_placeholder = payload.get("think", "").strip() == "Phân tích chi tiết..."
        verify_placeholder = payload.get("verify", "").strip() == "Quá trình kiểm chứng..."
        p0_quality[source_id] = {
            "think_status": "PLACEHOLDER_EXACT" if think_placeholder else "NON_PLACEHOLDER",
            "verify_status": "PLACEHOLDER_EXACT" if verify_placeholder else "NON_PLACEHOLDER",
            "substantive_think_verify": not (think_placeholder or verify_placeholder),
            "prediction_unchanged": True,
        }

    qa = {
        "artifact_type": "post_development_dependency_anchor_qa",
        "artifact_version": "1.0",
        "results_zip": str(args.results_zip.resolve()),
        "results_zip_sha256": sha256(args.results_zip.read_bytes()),
        "canonical_dataset_sha256": dataset_hash,
        "development_scope": "DEVELOPMENT_ONLY_NOT_OFFICIAL",
        "ground_truth_used_for_scoring": False,
        "experimental_llm_calls_in_this_task": 0,
        "historical_run_verification": {
            "jobs": len(result_objects), "all_completed": all_completed,
            "real_calls": calls, "received_events": len(received),
            "response_hash_matches": response_hash_matches,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "retry_or_error": retry_or_error,
        },
        "anchor_reports": anchor_reports,
        "p0_think_verify_quality": p0_quality,
        "interpretation_limit": "Anchor presence is textual support only; semantic dependency correctness is NOT_ASSESSED.",
    }
    write_new(args.output / "dependency_anchor_qa.json", json.dumps(qa, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    anchor_lines = []
    for sid, report in anchor_reports.items():
        for item in report["anchors"]:
            anchor_lines.append(
                f"| {sid} | `{item['anchor']}` | {item['status']} | {item['matched_diff_line_numbers'] or '-'} |"
            )
    evidence_report = f"""# Post-development evidence QA

## Scope and run integrity

The supplied development archive was inspected read-only. It contains 6/6 completed jobs and 16/16 received calls; all 16 raw-response hashes match. Recorded usage is 98,979 input and 3,628 output tokens, with no retry or technical error. This task made **ZERO** LLM/Ollama/API calls and did not modify predictions, responses, or ground truth.

## P2 dependency-anchor QA

| source | anchor | textual status | diff line(s) |
|---|---|---|---:|
{chr(10).join(anchor_lines)}

`EXACT_MATCH` means byte-for-text line equality after newline splitting. `WHITESPACE_NORMALIZED_MATCH` is separately reported and is not exact. `UNSUPPORTED` means the anchor is absent even after whitespace normalization. Text presence never proves that the stated dependency type or security interpretation is correct.

Finding: 18922 has one whitespace-normalized true diff anchor and one unsupported invented include anchor. 18867 has two repeated whitespace-normalized true deletion anchors. Thus P2 evidence includes useful textual references but also at least one unsupported claim; semantic correctness remains human-review work.

## 18867 reasoning disagreement

The authoritative patch removes the Salsa20 fast path in both implementations because it can call `blkcipher_walk_done()` when zero bytes were provided. P0 and P1 instead state that no relevant change exists and predict `NOT_PATCHED` with HIGH confidence; P2 abstains as `NOT_SURE`. Because the canonical label is still `PENDING`, this is a **reasoning disagreement**, not a proven model error against gold ground truth.

P0's `think` and `verify` fields for both 18867 and 18922 exactly repeat the output-schema placeholders (`Phân tích chi tiết...`, `Quá trình kiểm chứng...`). The separate `reasoning` field is nonempty, but the Think & Verify trace is not substantive. P1 local responses for 18867 also incorrectly describe the two changed files as unchanged despite their diffs.

## Prompt disposition

Baseline prompt hashes are preserved. Do not revise P0/P1/P2 based on two development examples. Proposed **vNext only**: require evidence-bearing `think`/`verify` content, reject exact placeholder echoes during output QA, and ask local analysis to cite at least one changed line when a nonempty diff is supplied. Any vNext must be separately versioned and evaluated; no silent baseline change is authorized.
"""
    write_new(args.output / "post_development_evidence_qa.md", evidence_report)

    packets = [
        "# Priority reviewer packet", "",
        "Mỗi mục dưới đây là hồ sơ ngắn hỗ trợ quyết định; không mục nào là APPROVED/VERIFIED. Hai ID development-exposed không được dùng làm held-out.", "",
    ]
    resource_totals = 0
    resource_matches = 0
    for source_id in PRIORITY_IDS:
        record = records[source_id]
        retrieval = args.evidence_root / source_id / "retrieval"
        manifest_path = retrieval / "retrieval_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matches = 0
        for resource in manifest["resources"]:
            resource_totals += 1
            path = retrieval / resource["local_path"]
            if path.is_file() and sha256(path.read_bytes()) == resource["sha256"]:
                resource_matches += 1
                matches += 1
        diff_resource = next(item for item in manifest["resources"] if item["kind"] == "ORIGINAL_DIFF")
        paths = ", ".join(
            (item.get("path_after") or item.get("path_before") or "UNKNOWN")
            for item in manifest["operations"]
        )
        exposed = source_id in EXPOSED_IDS
        uncertainty = (
            "Development-exposed; ground truth/security review PENDING; exclude from held-out."
            if exposed else
            "Ground truth/security review PENDING; derived change-count gap requires per-sample decision."
        )
        packets += [
            f"## {source_id} — {record['cve_id']}", "",
            f"- Commit/parent: `{record['fix_commit']}` / `{record['parent_commit']}`.",
            f"- Changed files ({record['changed_file_count']}): {paths}.",
            f"- Patch purpose from commit/diff: {PATCH_SUMMARIES[source_id]}",
            f"- Evidence: [{record['repository']} commit]({record['commit_url']}); original patch SHA-256 `{diff_resource['sha256']}`; manifest SHA-256 `{sha256(manifest_path.read_bytes())}`; resources {matches}/{len(manifest['resources'])} match.",
            f"- Context: historical full-context static P0/P1/P2 = {'FIT' if exact_by_id[source_id]['all_static_prompts_fit'] else 'NOT_FIT'}; runtime P1/P2 guard still required.",
            f"- Uncertainty: {uncertainty}",
            "- Current status: `PENDING`; researcher must decide APPROVE/DEFER/REJECT with identity, UTC timestamp, rationale and evidence refs.", "",
        ]
    packets += [
        "## Integrity summary", "",
        f"Priority evidence resources: {resource_matches}/{resource_totals} SHA-256 matches. Canonical dataset SHA-256 `{dataset_hash}` matches exact-token report. No historical artifact was rewritten.", "",
    ]
    write_new(args.output / "priority_reviewer_packet.md", "\n".join(packets))

    unexposed = [sid for sid in PRIORITY_IDS if sid not in EXPOSED_IDS]
    decision = f"""# Official-pilot feasibility decision

## Decision: NO-GO under the current 20–40 pilot target

- Historical full-context exact static fit for all P0/P1/P2: 6/20 (`{', '.join(PRIORITY_IDS)}`).
- Development-exposed and therefore excluded from held-out: 2 (`18922`, `18867`).
- Unexposed context-fit technical candidates: 4 (`{', '.join(unexposed)}`).
- Those four still have `PENDING` security/ground-truth review and derived-only waiver decisions; currently eligible held-out samples: **0**.
- Shortfall today versus minimum pilot 20: **20 samples**. Even if all four pass human review, shortfall remains **16**.

## Realistic November scope

With current evidence, the defensible scope is: retain 18922/18867 as development-only qualitative cases, and prepare at most four unexposed held-out positive samples after attributable review. This is insufficient for the proposal's 20–40 pilot and far below a 100–150 main cohort. Do not report Accuracy/F1 or treat PENDING fix-commit records as verified gold.

## Critical path

1. Researcher reviews the four unexposed packets first and decides each derived-only waiver/security scope.
2. Freeze exposed-family exclusions and an ordered list only after decisions.
3. If a larger cohort remains mandatory, screen a small targeted replacement batch for historical full-context fit before expensive security review; do not bulk-canonicalize hundreds.
4. Send the existing `gvhd_scope_adjustment_draft.md` now. It should be updated only with the verified current counts above and must remain DRAFT until GVHD responds.
5. Official inference remains blocked until review records, manifest, context gates and explicit cohort authorization are all present.

## Recommended conclusion

An evidence-valid reduced pilot can plausibly be prepared during November, but the originally proposed 20–40 official pilot cannot currently be supported. The scientifically honest next step is simultaneous four-sample review plus GVHD scope-amendment, not relaxing gates.
"""
    write_new(args.output / "pilot_feasibility_decision.md", decision)


if __name__ == "__main__":
    main()
