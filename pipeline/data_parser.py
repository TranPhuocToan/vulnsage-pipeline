"""
pipeline/data_parser.py – MODULE 1: Tiền xử lý Dữ liệu VulnSage G3.

Lý thuyết áp dụng:
    "Lost in the End" (Sovrano et al., FSE 2025):
    - KHÔNG cắt code theo số token cố định (Invariant I4)
    - Phải giữ nguyên vẹn ranh giới hàm C/C++
    - Mỗi file trong G3 được xử lý độc lập, giữ nguyên vẹn

Chức năng:
    - Parse raw VulnSage G3 JSON → structured per-file format
    - Trích xuất changed_lines (điểm neo cho SySeVR/P2)
    - Validate không cắt ngang hàm C/C++ (brace balance)
    - Tạo pilot subset (20 mẫu) với round-robin theo CWE

Input:  data/raw/vulnsage_g3.json
Output: data/processed/parsed_g3.json, data/pilot/pilot_20.json

References:
    - Phần 5 – PIPELINE_ROADMAP.md
    - Invariant I4: KHÔNG cắt hàm C/C++ ở giữa
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_RAW_PATH, DATA_PATH, PILOT_PATH, PILOT_SIZE


def parse_vulnsage_g3(raw_data_path: str, output_path: str) -> List[Dict[str, Any]]:
    """
    Parse raw VulnSage G3 dataset thành format chuẩn hóa.

    Quy trình:
    1. Đọc raw JSON → lọc chỉ lấy mẫu G3
    2. Với mỗi mẫu: parse từng file, trích xuất changed_lines
    3. Validate: kiểm tra brace balance (Invariant I4)
    4. Ghi output JSON

    Args:
        raw_data_path: Đường dẫn đến file raw VulnSage G3 JSON.
        output_path: Đường dẫn output cho parsed JSON.

    Returns:
        List[dict]: Danh sách các mẫu G3 đã parse.

    Schema output (mỗi phần tử):
        {
            "sample_id", "cve_id", "cwe_id", "label", "noise_score",
            "commit_hash", "files": [{
                "file_name", "vulnerable_code", "patched_code",
                "git_diff", "changed_lines", "file_index", "total_files"
            }],
            "full_git_diff", "metadata": {
                "project", "num_files", "num_changed_lines"
            }
        }
    """
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_samples: List[Dict] = json.load(f)

    parsed: List[Dict[str, Any]] = []
    skipped_count: int = 0
    warning_count: int = 0

    for sample in raw_samples:
        # Lọc chỉ lấy mẫu G3 (multi-function, multi-file)
        if sample.get("granularity") != "G3":
            skipped_count += 1
            continue

        parsed_sample: Dict[str, Any] = {
            "sample_id": sample["id"],
            "cve_id": sample.get("cve_id", ""),
            "cwe_id": sample.get("cwe_id", ""),
            "label": sample["label"],
            "noise_score": sample.get("noise_score", 0.0),
            "commit_hash": sample.get("commit_hash", ""),
            "files": [],
            "full_git_diff": sample.get("full_git_diff", ""),
            "metadata": {},
        }

        total_files: int = len(sample["files"])

        for idx, file_data in enumerate(sample["files"]):
            # Trích xuất changed_lines = "điểm neo" (Candidate) theo SySeVR
            changed_lines: List[Dict[str, str]] = extract_changed_lines(
                file_data.get("git_diff", "")
            )

            parsed_file: Dict[str, Any] = {
                "file_name": file_data["file_name"],
                "vulnerable_code": file_data.get("vulnerable_code", ""),
                "patched_code": file_data.get("patched_code", ""),
                "git_diff": file_data.get("git_diff", ""),
                "changed_lines": changed_lines,
                "file_index": idx,
                "total_files": total_files,
            }

            # Invariant I4: KIỂM TRA không cắt ngang hàm C/C++
            if not validate_no_function_truncation(
                parsed_file["vulnerable_code"],
                file_name=file_data["file_name"],
                sample_id=sample["id"],
            ):
                warning_count += 1

            if not validate_no_function_truncation(
                parsed_file["patched_code"],
                file_name=file_data["file_name"],
                sample_id=sample["id"],
                code_type="patched",
            ):
                warning_count += 1

            parsed_sample["files"].append(parsed_file)

        # Metadata tổng hợp
        parsed_sample["metadata"] = {
            "project": sample.get("source", ""),
            "num_files": total_files,
            "num_changed_lines": sum(
                len(f["changed_lines"]) for f in parsed_sample["files"]
            ),
        }
        parsed.append(parsed_sample)

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    print(f"[data_parser] Parsed {len(parsed)} G3 samples → {output_path}")
    print(f"[data_parser] Skipped {skipped_count} non-G3 samples")
    if warning_count > 0:
        print(f"[data_parser] ⚠ {warning_count} brace-balance warnings")

    return parsed


def extract_changed_lines(git_diff: str) -> List[Dict[str, str]]:
    """
    Trích xuất các dòng có dấu + và - từ git diff.

    Đây là "điểm neo" (Candidate) theo thuật ngữ SySeVR.
    KHÔNG bao gồm dòng @@ header và +++ / --- file markers.

    Args:
        git_diff: Chuỗi git diff của một file.

    Returns:
        List[dict]: Mỗi dict có 'type' ('added'|'removed') và 'content'.

    Example:
        >>> extract_changed_lines("+  kfree(sk);\\n-  // missing free")
        [{'type': 'added', 'content': 'kfree(sk);'},
         {'type': 'removed', 'content': '// missing free'}]
    """
    changed: List[Dict[str, str]] = []

    if not git_diff:
        return changed

    for line in git_diff.split("\n"):
        # Bỏ qua file markers (+++ / ---) và hunk headers (@@)
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            continue

        if line.startswith("+"):
            content = line[1:].strip()
            if content:  # Bỏ qua dòng trống
                changed.append({"type": "added", "content": content})
        elif line.startswith("-"):
            content = line[1:].strip()
            if content:  # Bỏ qua dòng trống
                changed.append({"type": "removed", "content": content})

    return changed


def validate_no_function_truncation(
    code: str,
    file_name: str = "",
    sample_id: str = "",
    code_type: str = "vulnerable",
) -> bool:
    """
    Kiểm tra code không bị cắt ngang giữa hàm C/C++.

    Nguyên lý từ "Lost in the End" (Sovrano et al., FSE 2025):
    KHÔNG cắt theo token cố định. Giữ nguyên vẹn ranh giới hàm.

    Heuristic: Đếm { } – nếu mất cân bằng → cảnh báo (Invariant I4).
    Loại trừ { } trong string literals và comments để giảm false positive.

    Args:
        code: Mã nguồn C/C++ cần kiểm tra.
        file_name: Tên file (cho log message).
        sample_id: ID mẫu (cho log message).
        code_type: Loại code ('vulnerable' hoặc 'patched').

    Returns:
        bool: True nếu brace balanced, False nếu mất cân bằng.
    """
    if not code:
        return True

    # Loại bỏ string literals và comments trước khi đếm braces
    cleaned = _remove_strings_and_comments(code)
    open_braces: int = cleaned.count("{")
    close_braces: int = cleaned.count("}")

    if open_braces != close_braces:
        print(
            f"[WARNING] Possible truncation in {code_type} code: "
            f"sample={sample_id}, file={file_name}, "
            f"{{ {open_braces} vs }} {close_braces}"
        )
        return False

    return True


def _remove_strings_and_comments(code: str) -> str:
    """
    Loại bỏ string literals và comments khỏi code C/C++
    để đếm brace chính xác hơn.

    Xử lý:
    - Single-line comments: // ...
    - Multi-line comments: /* ... */
    - String literals: "..." (với escaped quotes)
    - Character literals: '...'

    Args:
        code: Mã nguồn C/C++ gốc.

    Returns:
        str: Code đã loại bỏ strings và comments.
    """
    # Pattern loại bỏ theo thứ tự ưu tiên
    pattern = re.compile(
        r'//.*?$'           # Single-line comment
        r'|/\*.*?\*/'       # Multi-line comment
        r'|"(?:\\.|[^"\\])*"'  # String literal
        r"|'(?:\\.|[^'\\])*'",  # Character literal
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub("", code)


def create_pilot_subset(
    parsed_data: List[Dict[str, Any]],
    pilot_size: int = PILOT_SIZE,
    output_path: str = PILOT_PATH,
) -> List[Dict[str, Any]]:
    """
    Tạo tập pilot với đa dạng CWE (round-robin theo CWE type).

    Chiến lược: Round-robin qua các CWE categories để đảm bảo
    tập pilot đại diện cho đa dạng loại lỗ hổng.

    Args:
        parsed_data: Danh sách mẫu G3 đã parse.
        pilot_size: Số mẫu trong tập pilot (mặc định: 20).
        output_path: Đường dẫn output cho pilot JSON.

    Returns:
        List[dict]: Danh sách mẫu pilot.
    """
    # Nhóm theo CWE
    by_cwe: Dict[str, List[Dict]] = defaultdict(list)
    for sample in parsed_data:
        cwe_key = sample.get("cwe_id", "UNKNOWN")
        by_cwe[cwe_key].append(sample)

    # Round-robin chọn mẫu
    pilot: List[Dict[str, Any]] = []
    round_robin_idx: Dict[str, int] = {k: 0 for k in by_cwe}
    cwe_keys: List[str] = sorted(by_cwe.keys())

    while len(pilot) < pilot_size and cwe_keys:
        exhausted_cwes: List[str] = []
        for cwe in cwe_keys:
            if len(pilot) >= pilot_size:
                break
            idx = round_robin_idx[cwe]
            if idx < len(by_cwe[cwe]):
                pilot.append(by_cwe[cwe][idx])
                round_robin_idx[cwe] = idx + 1
            else:
                exhausted_cwes.append(cwe)

        # Loại bỏ CWE đã hết mẫu
        for cwe in exhausted_cwes:
            cwe_keys.remove(cwe)

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pilot, f, indent=2, ensure_ascii=False)

    # Thống kê pilot
    pilot_cwes = defaultdict(int)
    for s in pilot:
        pilot_cwes[s.get("cwe_id", "UNKNOWN")] += 1

    print(f"[data_parser] Created pilot subset: {len(pilot)} samples → {output_path}")
    print(f"[data_parser] CWE distribution: {dict(pilot_cwes)}")

    return pilot


def main() -> None:
    """
    Entry point cho data_parser module.

    Sử dụng:
        python -m pipeline.data_parser
        python -m pipeline.data_parser --raw data/raw/vulnsage_g3.json
        python -m pipeline.data_parser --pilot-only
    """
    parser = argparse.ArgumentParser(
        description="VulnSage G3 Data Parser – Tiền xử lý dữ liệu"
    )
    parser.add_argument(
        "--raw",
        type=str,
        default=DATA_RAW_PATH,
        help=f"Đường dẫn raw dataset (mặc định: {DATA_RAW_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DATA_PATH,
        help=f"Đường dẫn output parsed (mặc định: {DATA_PATH})",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=PILOT_SIZE,
        help=f"Số mẫu pilot (mặc định: {PILOT_SIZE})",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Chỉ tạo pilot subset từ parsed data đã có",
    )
    args = parser.parse_args()

    if args.pilot_only:
        # Chỉ tạo pilot từ parsed data đã tồn tại
        if not os.path.exists(args.output):
            print(f"[ERROR] Parsed data not found: {args.output}")
            print("Run without --pilot-only first to parse raw data.")
            sys.exit(1)

        with open(args.output, "r", encoding="utf-8") as f:
            parsed_data = json.load(f)

        create_pilot_subset(parsed_data, pilot_size=args.pilot_size)
    else:
        # Full pipeline: parse raw → create pilot
        if not os.path.exists(args.raw):
            print(f"[ERROR] Raw data not found: {args.raw}")
            print(f"Please place VulnSage G3 dataset at: {args.raw}")
            sys.exit(1)

        parsed_data = parse_vulnsage_g3(args.raw, args.output)
        create_pilot_subset(parsed_data, pilot_size=args.pilot_size)

    print("[data_parser] ✅ Done!")


if __name__ == "__main__":
    main()
