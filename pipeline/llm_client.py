"""
pipeline/llm_client.py – Module gọi API LLM dùng chung cho P0, P1, P2.

Module này cung cấp các hàm tiện ích để:
- Gọi LLM API qua OpenAI-compatible interface
- Retry với exponential backoff khi lỗi format hoặc API error (Invariant I5)
- Trích xuất JSON từ LLM response (xử lý markdown code fences)
- Parse prediction verdict và local analysis results
- Cộng dồn token usage giữa các bước pipeline

Invariants tuân thủ:
- I1: temperature = 0.0 trên MỌI lệnh gọi LLM
- I2: Mọi dữ liệu trung gian phải lưu dạng JSON
- I5: Retry max 3 lần → NOT_SURE nếu hết retry
- I6: Lưu log đầy đủ: prediction, ground_truth, tokens, time

References:
- Phụ lục A – PIPELINE_ROADMAP.md
- VulnSage (Zibaeirad & Vieira, 2025)
"""

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from config import (
    MODEL_NAME,
    API_KEY,
    API_BASE_URL,
    TEMPERATURE,
    MAX_OUTPUT_TOKENS,
    MAX_RETRY,
)


def get_llm_client() -> OpenAI:
    """
    Tạo và trả về OpenAI client instance.

    Hỗ trợ cả OpenAI gốc và các API tương thích (Gemini, Qwen, v.v.)
    thông qua API_BASE_URL.

    Returns:
        OpenAI: Client instance đã cấu hình.
    """
    return OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL if API_BASE_URL else None,
    )


def call_llm(prompt: str) -> Tuple[str, Dict[str, int]]:
    """
    Gọi LLM API một lần. Hỗ trợ OpenAI-compatible API.

    Args:
        prompt: Nội dung prompt gửi đến LLM.

    Returns:
        Tuple gồm:
            - response_text (str): Nội dung trả lời từ LLM.
            - token_usage (dict): Dictionary chứa input_tokens, output_tokens,
              total_tokens.

    Raises:
        Exception: Khi API call thất bại (network, auth, rate limit, v.v.)

    Note:
        Invariant I1: temperature LUÔN = 0.0 để đảm bảo tính tất định.
    """
    client = get_llm_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,  # Invariant I1: LUÔN = 0.0
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    text: str = response.choices[0].message.content
    usage: Dict[str, int] = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return text, usage


def call_llm_with_retry(
    prompt: str, max_retry: int = MAX_RETRY
) -> Tuple[str, Dict[str, int]]:
    """
    Gọi LLM với cơ chế retry khi lỗi format hoặc API error.

    Logic retry (Invariant I5):
    1. Gọi LLM → nhận response
    2. Trích xuất JSON từ response
    3. Kiểm tra có field 'verdict' hợp lệ
    4. Nếu lỗi → retry với exponential backoff (2^attempt giây)
    5. Sau max_retry lần → trả về fallback NOT_SURE

    Args:
        prompt: Nội dung prompt gửi đến LLM.
        max_retry: Số lần retry tối đa (mặc định từ config).

    Returns:
        Tuple gồm:
            - response_text (str): Response hợp lệ hoặc fallback JSON.
            - token_usage (dict): Token usage tích lũy.
    """
    for attempt in range(max_retry):
        try:
            response, tokens = call_llm(prompt)
            # Kiểm tra response có chứa JSON hợp lệ với verdict
            json_str = extract_json_from_response(response)
            parsed = json.loads(json_str)
            assert "verdict" in parsed, "Missing 'verdict' field in response"
            assert parsed["verdict"] in {
                "PATCHED",
                "NOT_PATCHED",
                "NOT_SURE",
            }, f"Invalid verdict: {parsed['verdict']}"
            return response, tokens
        except Exception as e:
            print(f"[RETRY {attempt + 1}/{max_retry}] Error: {e}")
            if attempt < max_retry - 1:
                sleep_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                time.sleep(sleep_time)

    # Fallback: NOT_SURE sau khi hết retry (Invariant I5)
    fallback = json.dumps(
        {"verdict": "NOT_SURE", "error": "max_retry_exceeded"},
        ensure_ascii=False,
    )
    return fallback, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def call_llm_with_retry_local(
    prompt: str, max_retry: int = MAX_RETRY
) -> Tuple[str, Dict[str, int]]:
    """
    Gọi LLM với retry cho Local Analysis (P1/P2 Step 1).

    Khác với call_llm_with_retry: kiểm tra 'local_verdict' thay vì 'verdict',
    vì Local Analysis prompt trả về local_verdict.

    Args:
        prompt: Nội dung prompt gửi đến LLM.
        max_retry: Số lần retry tối đa.

    Returns:
        Tuple gồm response_text và token_usage.
    """
    for attempt in range(max_retry):
        try:
            response, tokens = call_llm(prompt)
            json_str = extract_json_from_response(response)
            parsed = json.loads(json_str)
            # Local Analysis trả về 'local_verdict' thay vì 'verdict'
            assert "local_verdict" in parsed, "Missing 'local_verdict' field"
            return response, tokens
        except Exception as e:
            print(f"[RETRY LOCAL {attempt + 1}/{max_retry}] Error: {e}")
            if attempt < max_retry - 1:
                time.sleep(2**attempt)

    fallback = json.dumps(
        {
            "local_verdict": "UNCLEAR",
            "key_observations": [],
            "error": "max_retry_exceeded",
        },
        ensure_ascii=False,
    )
    return fallback, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def call_llm_with_retry_dependency(
    prompt: str, max_retry: int = MAX_RETRY
) -> Tuple[str, Dict[str, int]]:
    """
    Gọi LLM với retry cho Dependency Evidence Extraction (P2 Step 2).

    Kiểm tra response có chứa 'dependencies' array.

    Args:
        prompt: Nội dung prompt gửi đến LLM.
        max_retry: Số lần retry tối đa.

    Returns:
        Tuple gồm response_text và token_usage.
    """
    for attempt in range(max_retry):
        try:
            response, tokens = call_llm(prompt)
            json_str = extract_json_from_response(response)
            parsed = json.loads(json_str)
            assert "dependencies" in parsed, "Missing 'dependencies' field"
            return response, tokens
        except Exception as e:
            print(f"[RETRY DEP {attempt + 1}/{max_retry}] Error: {e}")
            if attempt < max_retry - 1:
                time.sleep(2**attempt)

    fallback = json.dumps(
        {
            "dependency_analysis": "Analysis failed after max retries",
            "dependencies": [],
            "num_cross_file_dependencies": 0,
            "overall_coupling": "LOOSE",
            "error": "max_retry_exceeded",
        },
        ensure_ascii=False,
    )
    return fallback, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def extract_json_from_response(text: str) -> str:
    """
    Trích xuất JSON từ LLM response một cách linh hoạt và dẻo dai.

    Xử lý:
    1. Markdown code fences (```json ... ``` hoặc ``` ... ```)
    2. Thuật toán cân bằng ngoặc nhọn (balanced braces) để lấy đúng object {...} ngoài cùng
    3. Tự động dọn dẹp trailing commas (dấu phẩy thừa trước } hoặc ])

    Args:
        text: Raw response text từ LLM.

    Returns:
        str: Chuỗi JSON đã trích xuất và làm sạch.

    Raises:
        ValueError: Khi không tìm thấy JSON hợp lệ trong response.
    """
    if not text or not isinstance(text, str):
        raise ValueError("Empty or invalid response text")

    # 1. Nếu có markdown code fence, ưu tiên lấy nội dung bên trong fence trước
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    candidate_text = fence_match.group(1) if fence_match else text

    # 2. Tìm dấu mở ngoặc '{' đầu tiên
    start_idx = candidate_text.find("{")
    if start_idx == -1:
        # Thử tìm trên toàn bộ văn bản gốc nếu trong fence không thấy
        start_idx = text.find("{")
        candidate_text = text
        if start_idx == -1:
            raise ValueError(f"No JSON object found in response: {text[:200]}...")

    # 3. Thuật toán Balanced Braces: duyệt tìm đúng dấu '}' đóng tương ứng
    balance = 0
    in_string = False
    escape = False
    end_idx = -1

    for i in range(start_idx, len(candidate_text)):
        char = candidate_text[i]
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
            continue
        elif not in_string:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance == 0:
                    end_idx = i + 1
                    break
        escape = False

    if end_idx != -1:
        raw_json = candidate_text[start_idx:end_idx]
    else:
        # Fallback: nếu LLM bị cắt cụt do max_tokens, lấy từ start đến dấu '}' cuối cùng
        last_brace = candidate_text.rfind("}")
        if last_brace != -1 and last_brace > start_idx:
            raw_json = candidate_text[start_idx:last_brace + 1]
        else:
            raise ValueError(f"Unbalanced braces in response: {text[:200]}...")

    # 4. Kiểm tra xem chuỗi JSON có parse được ngay không
    try:
        json.loads(raw_json)
        return raw_json
    except Exception:
        pass

    # 5. Dọn dẹp lỗi cú pháp phổ biến của LLM: Trailing commas (vd: {"a": 1,})
    sanitized = re.sub(r",\s*([\}\]])", r"\1", raw_json)
    try:
        json.loads(sanitized)
        return sanitized
    except Exception:
        pass

    return raw_json


def parse_prediction(response_text: str) -> str:
    """
    Trích xuất nhãn verdict từ response JSON.

    Args:
        response_text: Raw response text chứa JSON với field 'verdict'.

    Returns:
        str: Nhãn verdict ('PATCHED', 'NOT_PATCHED', hoặc 'NOT_SURE').
             Trả về 'NOT_SURE' nếu không parse được.
    """
    try:
        parsed = json.loads(extract_json_from_response(response_text))
        verdict = parsed.get("verdict", "NOT_SURE")
        if verdict in {"PATCHED", "NOT_PATCHED", "NOT_SURE"}:
            return verdict
        return "NOT_SURE"
    except Exception:
        return "NOT_SURE"


def parse_local_analysis(response_text: str) -> Dict[str, Any]:
    """
    Parse kết quả Local Analysis từ LLM response (P1/P2 Step 1).

    Args:
        response_text: Raw response text chứa JSON local analysis.

    Returns:
        dict: Dictionary chứa local_verdict, key_observations, v.v.
              Trả về dict mặc định với local_verdict='UNCLEAR' nếu parse lỗi.
    """
    try:
        return json.loads(extract_json_from_response(response_text))
    except Exception:
        return {
            "local_verdict": "UNCLEAR",
            "key_observations": [],
            "parse_error": True,
        }


def accumulate_tokens(
    total: Dict[str, int], new: Dict[str, int]
) -> None:
    """
    Cộng dồn token usage vào dictionary tổng.

    Hỗ trợ Invariant I6: Lưu log đầy đủ tokens cho báo cáo RQ3.

    Args:
        total: Dictionary tổng cần cập nhật (mutated in-place).
        new: Dictionary token usage mới cần cộng thêm.
    """
    for key in ["input_tokens", "output_tokens", "total_tokens"]:
        total[key] = total.get(key, 0) + new.get(key, 0)
