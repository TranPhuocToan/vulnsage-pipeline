"""
pipeline/pipeline_p0.py – MODULE 2: Pipeline P0 – Baseline Think & Verify.

Lý thuyết áp dụng:
    Think & Verify từ VulnSage paper (Zibaeirad & Vieira, 2025).

Mục đích:
    Điểm đối chứng (baseline). Tái hiện chính xác chiến lược P0
    của bài báo gốc: gom toàn bộ code đa tệp thành 1 context block,
    gửi 1 lần gọi LLM duy nhất.

Luồng xử lý:
    1. Gom toàn bộ code TẤT CẢ tệp → 1 CODE_BLOCK
    2. Gửi CODE_BLOCK + GIT_DIFF vào Think & Verify Prompt (1 LLM call)
    3. Parse response → nhãn JSON
    4. Retry nếu format sai (max 3 lần, Invariant I5)

Input:  data/processed/parsed_g3.json (hoặc data/pilot/pilot_20.json)
Output: results/main/p0_results.json (hoặc results/pilot/p0_pilot.json)

References:
    - Phần 6 – PIPELINE_ROADMAP.md
    - VulnSage paper: arXiv:2503.17885v1
    - Invariants: I1 (temp=0.0), I5 (retry→NOT_SURE), I6 (full logging)
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_RETRY
from pipeline.llm_client import call_llm_with_retry, parse_prediction


# Load prompt template
_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
    "p0_think_verify.txt",
)


def _load_prompt_template() -> str:
    """Load P0 Think & Verify prompt template từ file."""
    with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


THINK_VERIFY_PROMPT: str = _load_prompt_template()


def process_canonical_sample_p0(adapted, client, policy=None):
    """Phase-3B canonical entry point; client injection keeps tests offline."""
    from pipeline.offline_integration import run_canonical_sample
    return run_canonical_sample("p0", adapted, client, policy)


def run_p0(parsed_data_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Chạy Pipeline P0 (Baseline Think & Verify) trên toàn bộ dataset.

    Xử lý tuần tự từng mẫu, lưu incremental sau mỗi mẫu
    để tránh mất dữ liệu khi gián đoạn.

    Args:
        parsed_data_path: Đường dẫn đến parsed G3 JSON.
        output_path: Đường dẫn output cho P0 results.

    Returns:
        List[dict]: Danh sách kết quả P0 cho từng mẫu.
    """
    with open(parsed_data_path, "r", encoding="utf-8") as f:
        samples: List[Dict] = json.load(f)

    # Kiểm tra đã có kết quả trước đó (resume support)
    results: List[Dict[str, Any]] = []
    processed_ids: set = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
            results = existing
            processed_ids = {r["sample_id"] for r in existing}
            print(f"[P0] Resuming: {len(processed_ids)} samples already processed")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total = len(samples)
    for i, sample in enumerate(samples, 1):
        # Skip đã xử lý (resume)
        if sample["sample_id"] in processed_ids:
            continue

        print(f"[P0] Processing {i}/{total}: {sample['sample_id']} "
              f"(CVE: {sample['cve_id']}, {sample['metadata']['num_files']} files)")

        result = process_single_sample_p0(sample)
        results.append(result)

        # Lưu incremental (Invariant I2: JSON format)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  → Prediction: {result['prediction']} | "
              f"Tokens: {result['token_usage']['total_tokens']} | "
              f"Time: {result['inference_time_sec']:.2f}s")

    print(f"[P0] ✅ Completed: {len(results)} samples → {output_path}")
    return results


def process_single_sample_p0(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xử lý một mẫu G3 với Pipeline P0 (Think & Verify).

    Tái hiện chiến lược baseline từ VulnSage paper:
    - Gom toàn bộ code đa tệp thành 1 code block
    - 1 lần gọi LLM duy nhất

    Args:
        sample: Một mẫu G3 đã parse (từ parsed_g3.json).

    Returns:
        dict: Kết quả P0 bao gồm prediction, token_usage,
              inference_time_sec, raw_response, v.v.
              Invariant I6: Log đầy đủ cho báo cáo.
    """
    # Gom toàn bộ code thành 1 text block
    full_code_block: str = build_full_context_block(sample)

    # Format prompt P0
    prompt: str = THINK_VERIFY_PROMPT
    prompt = prompt.replace("{cve_id}", str(sample.get("cve_id", "")))
    prompt = prompt.replace("{cwe_id}", str(sample.get("cwe_id", "")))
    prompt = prompt.replace("{code_block}", full_code_block)
    prompt = prompt.replace("{git_diff}", sample.get("full_git_diff", ""))

    # Gọi LLM với retry (Invariant I5)
    start_time: float = time.time()
    llm_response, token_usage = call_llm_with_retry(prompt)
    inference_time: float = time.time() - start_time

    # Invariant I6: Log đầy đủ
    return {
        "sample_id": sample["sample_id"],
        "pipeline": "P0",
        "prediction": parse_prediction(llm_response),
        "ground_truth": sample["label"],
        "raw_response": llm_response,
        "token_usage": token_usage,
        "inference_time_sec": round(inference_time, 3),
        "num_llm_calls": 1,
        "cve_id": sample["cve_id"],
        "cwe_id": sample["cwe_id"],
        "num_files": sample["metadata"]["num_files"],
    }


def build_full_context_block(sample: Dict[str, Any]) -> str:
    """
    Gom toàn bộ code của tất cả file trong G3 thành 1 text block.

    Đây là đặc trưng của P0: xử lý toàn bộ ngữ cảnh đa tệp
    trong MỘT chu trình suy luận (hạn chế đã nêu trong VulnSage paper).

    Args:
        sample: Một mẫu G3 đã parse.

    Returns:
        str: Text block gồm tất cả file code.
    """
    blocks: List[str] = []
    for file_data in sample["files"]:
        block = (
            f"=== FILE: {file_data['file_name']} ===\n"
            f"--- VULNERABLE CODE ---\n{file_data['vulnerable_code']}\n\n"
            f"--- PATCHED CODE ---\n{file_data['patched_code']}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def main() -> None:
    """
    Entry point cho Pipeline P0.

    Sử dụng:
        python -m pipeline.pipeline_p0
        python -m pipeline.pipeline_p0 --input data/pilot/pilot_20.json --output results/pilot/p0_pilot.json
    """
    parser = argparse.ArgumentParser(
        description="Pipeline P0 – Baseline Think & Verify"
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
        default="results/main/p0_results.json",
        help="Đường dẫn output results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    run_p0(args.input, args.output)


if __name__ == "__main__":
    main()
