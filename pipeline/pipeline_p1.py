"""
pipeline/pipeline_p1.py – MODULE 3: Pipeline P1 – Phân rã Cơ bản.

Lý thuyết áp dụng:
    CAHM – Context-Aware Hierarchical Merging (Ou & Lapata, ACL 2025).

Nguyên lý:
    Chia nhỏ theo file → phân tích cục bộ độc lập (Level-0) →
    merge kết quả (Hierarchical Merging) → verdict cuối.

Ánh xạ CAHM → P1:
    | CAHM                  | P1 (đề tài)                    |
    |-----------------------|---------------------------------|
    | Chunk documents       | Tách G3 theo từng tệp          |
    | Level-0 summary       | Local Analysis per file         |
    | Recursive merge       | Global Synthesis (all locals)   |
    | Context propagation   | local_analysis_results list     |

Luồng xử lý:
    BƯỚC 1 – LOCAL ANALYZER: for each file → independent analysis
    BƯỚC 2 – GLOBAL SYNTHESIZER: merge all local results → final verdict

Input:  data/processed/parsed_g3.json
Output: results/main/p1_results.json + results/intermediate/p1_local_*.json

References:
    - Phần 7 – PIPELINE_ROADMAP.md
    - CAHM paper: hier_summ.py (Ou & Lapata, ACL 2025)
    - Invariants: I1, I2, I5, I6
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_INTERMEDIATE_DIR
from pipeline.llm_client import (
    call_llm_with_retry,
    call_llm_with_retry_local,
    parse_prediction,
    parse_local_analysis,
    accumulate_tokens,
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


LOCAL_ANALYSIS_PROMPT: str = _load_prompt("p1_local_analysis.txt")
GLOBAL_SYNTHESIS_PROMPT: str = _load_prompt("p1_global_synthesis.txt")


def run_p1(parsed_data_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Chạy Pipeline P1 (Phân rã Cơ bản – CAHM-inspired) trên toàn bộ dataset.

    Xử lý tuần tự từng mẫu với 2 bước per sample:
    1. Local Analysis (per file) – CAHM Level-0
    2. Global Synthesis (merge all) – CAHM Hierarchical Merge

    Args:
        parsed_data_path: Đường dẫn đến parsed G3 JSON.
        output_path: Đường dẫn output cho P1 results.

    Returns:
        List[dict]: Danh sách kết quả P1 cho từng mẫu.
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
            print(f"[P1] Resuming: {len(processed_ids)} samples already processed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = len(samples)
    for i, sample in enumerate(samples, 1):
        if sample["sample_id"] in processed_ids:
            continue

        print(f"[P1] Processing {i}/{total}: {sample['sample_id']} "
              f"(CVE: {sample['cve_id']}, {sample['metadata']['num_files']} files)")

        result = process_single_sample_p1(sample)
        results.append(result)

        # Lưu incremental (Invariant I2)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  → Prediction: {result['prediction']} | "
              f"LLM calls: {result['num_llm_calls']} | "
              f"Tokens: {result['token_usage']['total_tokens']} | "
              f"Time: {result['inference_time_sec']:.2f}s")

    print(f"[P1] ✅ Completed: {len(results)} samples → {output_path}")
    return results


def process_single_sample_p1(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xử lý một mẫu G3 với Pipeline P1 (CAHM 2-step).

    P1: Local Analysis (per file) → Global Synthesis.
    Áp dụng nguyên lý Level-0 then Merge của CAHM.

    Args:
        sample: Một mẫu G3 đã parse (từ parsed_g3.json).

    Returns:
        dict: Kết quả P1 bao gồm local_analyses, prediction,
              token_usage, inference_time_sec, v.v.
    """
    total_token_usage: Dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    start_time: float = time.time()

    # ========== BƯỚC 1: LOCAL ANALYSIS ==========
    # CAHM: summarize_batch(chunks) ↔ phân tích từng file
    local_analyses: List[Dict[str, Any]] = _run_local_analysis(
        sample, total_token_usage
    )

    # Lưu trung gian (bắt buộc theo Invariant I2)
    _save_intermediate_local(sample["sample_id"], local_analyses, prefix="p1")

    # ========== BƯỚC 2: GLOBAL SYNTHESIS ==========
    # CAHM: recursive merge → final summary
    synthesis_input: str = format_local_analyses_for_synthesis(local_analyses)
    synthesis_prompt: str = GLOBAL_SYNTHESIS_PROMPT
    synthesis_prompt = synthesis_prompt.replace("{num_files}", str(sample["metadata"]["num_files"]))
    synthesis_prompt = synthesis_prompt.replace("{cve_id}", str(sample.get("cve_id", "")))
    synthesis_prompt = synthesis_prompt.replace("{cwe_id}", str(sample.get("cwe_id", "")))
    synthesis_prompt = synthesis_prompt.replace("{local_analyses_text}", synthesis_input)

    synthesis_response, syn_token_usage = call_llm_with_retry(synthesis_prompt)
    accumulate_tokens(total_token_usage, syn_token_usage)

    inference_time: float = time.time() - start_time

    # Invariant I6: Log đầy đủ
    return {
        "sample_id": sample["sample_id"],
        "pipeline": "P1",
        "prediction": parse_prediction(synthesis_response),
        "ground_truth": sample["label"],
        "local_analyses": local_analyses,
        "synthesis_response": synthesis_response,
        "token_usage": total_token_usage,
        "inference_time_sec": round(inference_time, 3),
        "num_llm_calls": len(sample["files"]) + 1,
        "cve_id": sample["cve_id"],
        "cwe_id": sample["cwe_id"],
        "num_files": sample["metadata"]["num_files"],
    }


def _run_local_analysis(
    sample: Dict[str, Any],
    total_token_usage: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Chạy Local Analysis (CAHM Level-0) cho tất cả file trong sample.

    Mỗi file được phân tích ĐỘC LẬP bởi LLM, không biết kết quả
    của các file khác.

    Args:
        sample: Mẫu G3 đã parse.
        total_token_usage: Dict token tổng (mutated in-place).

    Returns:
        List[dict]: Kết quả local analysis cho từng file.
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
            "token_usage": token_usage,
        })
        accumulate_tokens(total_token_usage, token_usage)

    return local_analyses


def _save_intermediate_local(
    sample_id: str,
    local_analyses: List[Dict[str, Any]],
    prefix: str = "p1",
) -> None:
    """
    Lưu kết quả local analysis trung gian (Invariant I2).

    Args:
        sample_id: ID mẫu.
        local_analyses: Kết quả local analysis.
        prefix: Prefix cho tên file ('p1' hoặc 'p2').
    """
    os.makedirs(RESULTS_INTERMEDIATE_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_INTERMEDIATE_DIR, f"{prefix}_local_{sample_id}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(local_analyses, f, indent=2, ensure_ascii=False)


def format_local_analyses_for_synthesis(
    local_analyses: List[Dict[str, Any]],
) -> str:
    """
    Format kết quả local analyses thành text cho Global Synthesis prompt.

    Tương đương merge_summaries() trong CAHM.

    Args:
        local_analyses: Danh sách kết quả local analysis.

    Returns:
        str: Text đã format sẵn cho synthesis prompt.
    """
    formatted: List[str] = []
    total: int = len(local_analyses)

    for i, analysis in enumerate(local_analyses, 1):
        formatted.append(
            f"=== BÁO CÁO TỆP {i}/{total}: {analysis['file_name']} ===\n"
            f"Nhận định cục bộ: {analysis.get('local_verdict', 'UNCLEAR')}\n"
            f"Phân tích:\n{analysis['analysis_text']}\n"
        )

    return "\n".join(formatted)


def main() -> None:
    """
    Entry point cho Pipeline P1.

    Sử dụng:
        python -m pipeline.pipeline_p1
        python -m pipeline.pipeline_p1 --input data/pilot/pilot_20.json --output results/pilot/p1_pilot.json
    """
    parser = argparse.ArgumentParser(
        description="Pipeline P1 – Decomposition Basic (CAHM-inspired)"
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
        default="results/main/p1_results.json",
        help="Đường dẫn output results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_p1(args.input, args.output)


if __name__ == "__main__":
    main()
