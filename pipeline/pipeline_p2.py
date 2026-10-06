"""
pipeline/pipeline_p2.py – MODULE 4: Pipeline P2 – Phân rã Nâng cao.

Lý thuyết áp dụng:
    SySeVR (Li et al., IEEE TDSC 2022) – Dependency Tracing concept.

Nguyên lý:
    Thêm bước trích xuất "Bằng chứng phụ thuộc" (Dependency Evidence)
    liên tệp trước khi tổng hợp. Sử dụng các dòng +/- trong git diff
    làm "Candidate Points" (điểm neo) theo thuật ngữ SySeVR.

Ánh xạ SySeVR → P2:
    | SySeVR                    | P2 (đề tài)                          |
    |---------------------------|---------------------------------------|
    | Candidate generation      | Dòng +/- = điểm neo                  |
    | Program Slice (AST/CFG)   | LLM truy vết cross-file dependencies  |
    | Semantic slice (data flow)| LLM liệt kê data dependencies        |
    | Vulnerability repr.       | dependency_evidence.json              |

Luồng xử lý (3 bước):
    BƯỚC 1 – LOCAL ANALYSIS (giống P1, tái sử dụng logic)
    BƯỚC 2 – DEPENDENCY EVIDENCE EXTRACTION (SySeVR-inspired)
             ⚠ Invariant I8: CHỈ dùng git_diff + metadata, KHÔNG raw code
    BƯỚC 3 – EVIDENCE-AUGMENTED SYNTHESIS

Input:  data/processed/parsed_g3.json
Output: results/main/p2_results.json + results/intermediate/p2_dep_*.json

References:
    - Phần 8 – PIPELINE_ROADMAP.md
    - SySeVR paper (Li et al., IEEE TDSC 2022)
    - Invariants: I1, I2, I5, I6, I8
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_INTERMEDIATE_DIR
from pipeline.llm_client import (
    call_llm_with_retry,
    call_llm_with_retry_local,
    call_llm_with_retry_dependency,
    parse_prediction,
    parse_local_analysis,
    accumulate_tokens,
)
from pipeline.pipeline_p1 import (
    LOCAL_ANALYSIS_PROMPT,
    format_local_analyses_for_synthesis,
    _save_intermediate_local,
)


# Load prompt templates
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
)


def _load_prompt(filename: str) -> str:
    """Load prompt template từ file."""
    path = os.path.join(_PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


DEPENDENCY_EXTRACT_PROMPT: str = _load_prompt("p2_dependency_extract.txt")
EVIDENCE_SYNTHESIS_PROMPT: str = _load_prompt("p2_evidence_synthesis.txt")


def run_p2(parsed_data_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Chạy Pipeline P2 (Phân rã Nâng cao – SySeVR-inspired) trên toàn bộ dataset.

    Xử lý tuần tự từng mẫu với 3 bước per sample:
    1. Local Analysis (per file) – tái sử dụng từ P1
    2. Dependency Evidence Extraction – SySeVR Candidate → Dependencies
    3. Evidence-Augmented Synthesis – kết hợp local + dependency → verdict

    Args:
        parsed_data_path: Đường dẫn đến parsed G3 JSON.
        output_path: Đường dẫn output cho P2 results.

    Returns:
        List[dict]: Danh sách kết quả P2 cho từng mẫu.
    """
    with open(parsed_data_path, "r", encoding="utf-8") as f:
        samples: List[Dict] = json.load(f)

    # Resume support
    results: List[Dict[str, Any]] = []
    processed_ids: set = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing
            processed_ids = {r["sample_id"] for r in existing}
            print(f"[P2] Resuming: {len(processed_ids)} samples already processed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = len(samples)
    for i, sample in enumerate(samples, 1):
        if sample["sample_id"] in processed_ids:
            continue

        print(f"[P2] Processing {i}/{total}: {sample['sample_id']} "
              f"(CVE: {sample['cve_id']}, {sample['metadata']['num_files']} files)")

        result = process_single_sample_p2(sample)
        results.append(result)

        # Lưu incremental (Invariant I2)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  → Prediction: {result['prediction']} | "
              f"LLM calls: {result['num_llm_calls']} | "
              f"Deps found: {len(result['dependency_evidence'].get('dependencies', []))} | "
              f"Tokens: {result['token_usage']['total_tokens']} | "
              f"Time: {result['inference_time_sec']:.2f}s")

    print(f"[P2] ✅ Completed: {len(results)} samples → {output_path}")
    return results


def process_single_sample_p2(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xử lý một mẫu G3 với Pipeline P2 (3-step SySeVR-inspired).

    P2: Local Analysis → Dependency Evidence Extraction → Evidence-Augmented Synthesis.

    Args:
        sample: Một mẫu G3 đã parse (từ parsed_g3.json).

    Returns:
        dict: Kết quả P2 bao gồm local_analyses, dependency_evidence,
              prediction, token_usage, inference_time_sec, v.v.
    """
    total_token_usage: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    start_time: float = time.time()

    # ========== BƯỚC 1: LOCAL ANALYSIS (tái sử dụng logic của P1) ==========
    local_analyses: List[Dict[str, Any]] = _run_local_analysis_p2(
        sample, total_token_usage
    )

    # Lưu trung gian local (Invariant I2)
    _save_intermediate_local(sample["sample_id"], local_analyses, prefix="p2")

    # ========== BƯỚC 2: DEPENDENCY EVIDENCE EXTRACTION ==========
    # SySeVR: dòng +/- = Candidate → truy vết Program Slice qua cross-file
    # ⚠ Invariant I8: CHỈ dùng git_diff và metadata, KHÔNG raw code
    dependency_evidence: Dict[str, Any] = _run_dependency_extraction(
        sample, total_token_usage
    )

    # Lưu dependency evidence riêng (cho Manual Verification sau)
    _save_dependency_evidence(sample["sample_id"], dependency_evidence)

    # ========== BƯỚC 3: EVIDENCE-AUGMENTED SYNTHESIS ==========
    synthesis_response: str = _run_evidence_synthesis(
        sample, local_analyses, dependency_evidence, total_token_usage
    )

    inference_time: float = time.time() - start_time

    # Invariant I6: Log đầy đủ
    return {
        "sample_id": sample["sample_id"],
        "pipeline": "P2",
        "prediction": parse_prediction(synthesis_response),
        "ground_truth": sample["label"],
        "local_analyses": local_analyses,
        "dependency_evidence": dependency_evidence,
        "synthesis_response": synthesis_response,
        "token_usage": total_token_usage,
        "inference_time_sec": round(inference_time, 3),
        "num_llm_calls": len(sample["files"]) + 2,  # local + dep + synthesis
        "cve_id": sample["cve_id"],
        "cwe_id": sample["cwe_id"],
        "num_files": sample["metadata"]["num_files"],
    }


def _run_local_analysis_p2(
    sample: Dict[str, Any],
    total_token_usage: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Chạy Local Analysis cho P2 (tái sử dụng P1 logic).

    Khác P1: thêm changed_lines vào kết quả (điểm neo SySeVR).

    Args:
        sample: Mẫu G3 đã parse.
        total_token_usage: Dict token tổng (mutated in-place).

    Returns:
        List[dict]: Kết quả local analysis + changed_lines cho từng file.
    """
    local_analyses: List[Dict[str, Any]] = []

    for file_data in sample["files"]:
        prompt: str = LOCAL_ANALYSIS_PROMPT
        prompt = prompt.replace("{file_name}", str(file_data["file_name"]))
        prompt = prompt.replace("{file_index}", str(file_data["file_index"] + 1))
        prompt = prompt.replace("{total_files}", str(file_data["total_files"]))
        prompt = prompt.replace("{cve_id}", str(sample.get("cve_id", "")))
        prompt = prompt.replace("{cwe_id}", str(sample.get("cwe_id", "")))
        prompt = prompt.replace("{vulnerable_code}", str(file_data["vulnerable_code"]))
        prompt = prompt.replace("{patched_code}", str(file_data["patched_code"]))
        prompt = prompt.replace("{git_diff}", str(file_data.get("git_diff", "")))

        response, token_usage = call_llm_with_retry_local(prompt)
        local_result: Dict = parse_local_analysis(response)

        local_analyses.append({
            "file_name": file_data["file_name"],
            "file_index": file_data["file_index"],
            "analysis_text": response,
            "local_verdict": local_result.get("local_verdict", "UNCLEAR"),
            "key_observations": local_result.get("key_observations", []),
            "changed_lines": file_data["changed_lines"],  # ← Điểm neo SySeVR
            "token_usage": token_usage,
        })
        accumulate_tokens(total_token_usage, token_usage)

    return local_analyses


def _run_dependency_extraction(
    sample: Dict[str, Any],
    total_token_usage: Dict[str, int],
) -> Dict[str, Any]:
    """
    Chạy Dependency Evidence Extraction (P2 Bước 2).

    SySeVR: dòng +/- = Candidate → truy vết cross-file dependencies.

    ⚠ INVARIANT I8: CHỈ dùng git_diff + metadata, KHÔNG dùng raw code
    để tránh nhiễu.

    Args:
        sample: Mẫu G3 đã parse.
        total_token_usage: Dict token tổng (mutated in-place).

    Returns:
        dict: Dependency evidence (dependencies list, coupling level, etc.)
    """
    dependency_prompt: str = DEPENDENCY_EXTRACT_PROMPT
    dependency_prompt = dependency_prompt.replace("{cve_id}", str(sample.get("cve_id", "")))
    dependency_prompt = dependency_prompt.replace("{cwe_id}", str(sample.get("cwe_id", "")))
    dependency_prompt = dependency_prompt.replace("{num_files}", str(sample["metadata"]["num_files"]))
    dependency_prompt = dependency_prompt.replace("{file_list}", format_file_list_for_dependency(sample))
    dependency_prompt = dependency_prompt.replace("{full_git_diff}", sample.get("full_git_diff", ""))

    dep_response, dep_token_usage = call_llm_with_retry_dependency(dependency_prompt)
    dependency_evidence: Dict[str, Any] = parse_dependency_evidence(dep_response)
    accumulate_tokens(total_token_usage, dep_token_usage)

    return dependency_evidence


def _run_evidence_synthesis(
    sample: Dict[str, Any],
    local_analyses: List[Dict[str, Any]],
    dependency_evidence: Dict[str, Any],
    total_token_usage: Dict[str, int],
) -> str:
    """
    Chạy Evidence-Augmented Synthesis (P2 Bước 3).

    Kết hợp local analyses + dependency evidence → verdict cuối.

    Args:
        sample: Mẫu G3 đã parse.
        local_analyses: Kết quả local analysis từ Bước 1.
        dependency_evidence: Bằng chứng phụ thuộc từ Bước 2.
        total_token_usage: Dict token tổng (mutated in-place).

    Returns:
        str: Raw synthesis response từ LLM.
    """
    local_text: str = format_local_analyses_for_synthesis(local_analyses)
    dependency_table: str = format_dependency_table(dependency_evidence)

    synthesis_prompt: str = EVIDENCE_SYNTHESIS_PROMPT
    synthesis_prompt = synthesis_prompt.replace("{cve_id}", str(sample.get("cve_id", "")))
    synthesis_prompt = synthesis_prompt.replace("{cwe_id}", str(sample.get("cwe_id", "")))
    synthesis_prompt = synthesis_prompt.replace("{num_files}", str(sample["metadata"]["num_files"]))
    synthesis_prompt = synthesis_prompt.replace("{local_analyses_text}", local_text)
    synthesis_prompt = synthesis_prompt.replace("{dependency_evidence_table}", dependency_table)

    synthesis_response, syn_token_usage = call_llm_with_retry(synthesis_prompt)
    accumulate_tokens(total_token_usage, syn_token_usage)

    return synthesis_response


def _save_dependency_evidence(
    sample_id: str,
    dependency_evidence: Dict[str, Any],
) -> None:
    """
    Lưu dependency evidence riêng (cho Manual Verification sau).

    Args:
        sample_id: ID mẫu.
        dependency_evidence: Kết quả dependency extraction.
    """
    os.makedirs(RESULTS_INTERMEDIATE_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_INTERMEDIATE_DIR, f"p2_dep_{sample_id}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dependency_evidence, f, indent=2, ensure_ascii=False)


def format_file_list_for_dependency(sample: Dict[str, Any]) -> str:
    """
    Tạo danh sách tệp với điểm neo (SySeVR Candidate Points).

    ⚠ Invariant I8: Chỉ sử dụng metadata và changed_lines summary,
    KHÔNG bao gồm raw code.

    Args:
        sample: Mẫu G3 đã parse.

    Returns:
        str: Formatted file list với anchor points.
    """
    lines: List[str] = []
    for f in sample["files"]:
        changed_summary: List[str] = [
            c["content"][:80] for c in f["changed_lines"][:5]
        ]
        lines.append(
            f"- {f['file_name']}: {len(f['changed_lines'])} dòng thay đổi"
        )
        for c in changed_summary:
            lines.append(f"  Điểm neo: {c}")
    return "\n".join(lines)


def parse_dependency_evidence(response: str) -> Dict[str, Any]:
    """
    Parse JSON dependency evidence từ LLM response.

    Xử lý nhiều trường hợp:
    1. JSON trong markdown code fence
    2. JSON trực tiếp trong text
    3. Fallback nếu parse thất bại

    Args:
        response: Raw LLM response chứa dependency evidence.

    Returns:
        dict: Parsed dependency evidence, hoặc dict với parse_status='FAILED'.
    """
    try:
        # Thử trích xuất JSON từ code fence trước
        fence_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
        )
        if fence_match:
            return json.loads(fence_match.group(1))

        # Thử tìm JSON trực tiếp
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"[WARNING] Failed to parse dependency evidence: {e}")

    return {
        "raw_response": response,
        "dependencies": [],
        "num_cross_file_dependencies": 0,
        "overall_coupling": "LOOSE",
        "parse_status": "FAILED",
    }


def format_dependency_table(dependency_evidence: Dict[str, Any]) -> str:
    """
    Định dạng bảng bằng chứng phụ thuộc cho Evidence-Augmented Synthesis prompt.

    Args:
        dependency_evidence: Kết quả từ dependency extraction.

    Returns:
        str: Formatted dependency table cho synthesis prompt.
    """
    deps: List[Dict] = dependency_evidence.get("dependencies", [])

    if not deps:
        return "Không tìm thấy phụ thuộc liên tệp rõ ràng."

    lines: List[str] = ["BẢNG BẰNG CHỨNG PHỤ THUỘC LIÊN TỆP:"]
    for dep in deps:
        lines.append(
            f"- [{dep.get('source_file', '?')}] → [{dep.get('target_file', '?')}]: "
            f"{dep.get('dependency_type', 'UNKNOWN')} – "
            f"{dep.get('description', 'N/A')}"
        )

    # Thêm coupling level nếu có
    coupling = dependency_evidence.get("overall_coupling", "")
    if coupling:
        lines.append(f"\nMức liên kết tổng thể: {coupling}")

    return "\n".join(lines)


def main() -> None:
    """
    Entry point cho Pipeline P2.

    Sử dụng:
        python -m pipeline.pipeline_p2
        python -m pipeline.pipeline_p2 --input data/pilot/pilot_20.json --output results/pilot/p2_pilot.json
    """
    parser = argparse.ArgumentParser(
        description="Pipeline P2 – Decomposition + Dependency Evidence (SySeVR-inspired)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/parsed_g3.json",
        help="Đường dẫn input parsed data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/main/p2_results.json",
        help="Đường dẫn output results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_p2(args.input, args.output)


if __name__ == "__main__":
    main()
