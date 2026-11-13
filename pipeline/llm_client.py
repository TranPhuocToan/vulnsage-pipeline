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
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # The strict mock path must work without the paid API SDK.
    OpenAI = None  # type: ignore[assignment,misc]

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
    if OpenAI is None:
        raise RuntimeError("openai package is not installed; use an injected offline transport")
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


# ---------------------------------------------------------------------------
# Phase 3B strict/offline-capable client.  The legacy API above is retained for
# backwards compatibility; new canonical runs use this interface so technical
# failures are never converted into a model NOT_SURE verdict.

Transport = Callable[[str], Tuple[str, Optional[Dict[str, Optional[int]]]]]


@dataclass(frozen=True)
class AttemptTelemetry:
    attempt: int
    status: str
    duration_sec: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class StructuredCallResult:
    status: str
    response_kind: str
    parsed: Optional[Dict[str, Any]]
    raw_response: Optional[str]
    model_abstained: bool
    attempts: List[Dict[str, Any]]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


_VERDICTS = {"PATCHED", "NOT_PATCHED", "NOT_SURE"}
_LOCAL_VERDICTS = {"CONTRIBUTES_TO_FIX", "NOT_RELATED", "UNCLEAR"}
_THINK_VERIFY_EXAMPLE_VALUES = {
    "think": {"Phân tích chi tiết..."},
    "verify": {"Quá trình kiểm chứng..."},
}


def _strict_string_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")


def assess_think_verify_payload(payload: Any) -> Dict[str, Any]:
    """Assess P0 reasoning fields without changing the model verdict.

    The P0 prompt contains short JSON example values for ``think`` and
    ``verify``. A model can copy those examples while still returning a valid
    verdict schema. This QA check reports missing, empty, and exact-placeholder
    fields, but never rewrites the response or promotes/demotes the verdict.
    """
    reasons: List[str] = []
    if not isinstance(payload, dict):
        return {
            "status": "INVALID_PAYLOAD",
            "reason_codes": ["THINK_VERIFY_PAYLOAD_NOT_OBJECT"],
        }
    for field in ("think", "verify"):
        value = payload.get(field)
        if not isinstance(value, str):
            reasons.append(f"{field.upper()}_MISSING_OR_NOT_STRING")
            continue
        normalized = value.strip()
        if not normalized:
            reasons.append(f"{field.upper()}_EMPTY")
        elif normalized in _THINK_VERIFY_EXAMPLE_VALUES[field]:
            reasons.append(f"{field.upper()}_PLACEHOLDER_EXACT")
    return {
        "status": "SUBSTANTIVE" if not reasons else "NON_SUBSTANTIVE",
        "reason_codes": reasons,
    }


def validate_structured_payload(payload: Any, response_kind: str) -> Dict[str, Any]:
    """Validate response types/enums without silently coercing values."""
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    if response_kind == "verdict":
        if payload.get("verdict") not in _VERDICTS:
            raise ValueError("verdict must be PATCHED, NOT_PATCHED, or NOT_SURE")
        if payload.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("confidence must be HIGH, MEDIUM, or LOW")
        if not isinstance(payload.get("reasoning"), str):
            raise ValueError("reasoning must be a string")
    elif response_kind == "local":
        if payload.get("local_verdict") not in _LOCAL_VERDICTS:
            raise ValueError("invalid local_verdict enum")
        if payload.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("confidence must be HIGH, MEDIUM, or LOW")
        for field in ("file_name", "function_summary", "changes_summary", "security_purpose"):
            if not isinstance(payload.get(field), str):
                raise ValueError(f"{field} must be a string")
        _strict_string_list(payload.get("key_observations"), "key_observations")
    elif response_kind == "dependency":
        dependencies = payload.get("dependencies")
        if not isinstance(dependencies, list):
            raise ValueError("dependencies must be an array")
        for index, dependency in enumerate(dependencies):
            if not isinstance(dependency, dict):
                raise ValueError(f"dependencies[{index}] must be an object")
            for key in ("source_file", "target_file", "dependency_type", "description"):
                if not isinstance(dependency.get(key), str):
                    raise ValueError(f"dependencies[{index}].{key} must be a string")
            if dependency["dependency_type"] not in {
                "FUNCTION_CALL", "DATA_FLOW", "STRUCT_USAGE", "HEADER_INCLUDE", "OTHER"
            }:
                raise ValueError(f"dependencies[{index}].dependency_type has invalid enum")
            _strict_string_list(dependency.get("anchor_lines"), f"dependencies[{index}].anchor_lines")
            if not isinstance(dependency.get("security_relevance"), str):
                raise ValueError(f"dependencies[{index}].security_relevance must be a string")
        if not isinstance(payload.get("dependency_analysis"), str):
            raise ValueError("dependency_analysis must be a string")
        if type(payload.get("num_cross_file_dependencies")) is not int:
            raise ValueError("num_cross_file_dependencies must be an integer")
        if payload.get("overall_coupling") not in {"TIGHT", "MODERATE", "LOOSE"}:
            raise ValueError("overall_coupling has invalid enum")
    else:
        raise ValueError(f"unknown response_kind: {response_kind}")
    return payload


class StructuredLLMClient:
    """Retrying client with an injected transport, suitable for offline mocks."""

    def __init__(
        self,
        transport: Transport,
        max_attempts: int = 3,
        retry_delays: Optional[List[float]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.transport = transport
        self.max_attempts = max_attempts
        self.retry_delays = retry_delays or [0.0] * max_attempts
        self.sleep = sleep
        self.clock = clock

    def invoke(self, prompt: str, response_kind: str) -> StructuredCallResult:
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            started = self.clock()
            usage: Optional[Dict[str, Optional[int]]] = None
            raw: Optional[str] = None
            try:
                raw, usage = self.transport(prompt)
                parsed = json.loads(extract_json_from_response(raw))
                validated = validate_structured_payload(parsed, response_kind)
                telemetry = AttemptTelemetry(
                    attempt=attempt,
                    status="OK",
                    duration_sec=max(0.0, self.clock() - started),
                    input_tokens=(usage or {}).get("input_tokens"),
                    output_tokens=(usage or {}).get("output_tokens"),
                    total_tokens=(usage or {}).get("total_tokens"),
                )
                attempts.append(asdict(telemetry))
                verdict = validated.get("verdict") or validated.get("local_verdict")
                return StructuredCallResult(
                    status="OK",
                    response_kind=response_kind,
                    parsed=validated,
                    raw_response=raw,
                    model_abstained=verdict in {"NOT_SURE", "UNCLEAR"},
                    attempts=attempts,
                )
            except Exception as exc:
                last_error = exc
                telemetry = AttemptTelemetry(
                    attempt=attempt,
                    status="ERROR",
                    duration_sec=max(0.0, self.clock() - started),
                    input_tokens=(usage or {}).get("input_tokens"),
                    output_tokens=(usage or {}).get("output_tokens"),
                    total_tokens=(usage or {}).get("total_tokens"),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                attempts.append(asdict(telemetry))
                if attempt < self.max_attempts:
                    delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                    if delay > 0:
                        self.sleep(delay)

        assert last_error is not None
        return StructuredCallResult(
            status="TECHNICAL_ERROR",
            response_kind=response_kind,
            parsed=None,
            raw_response=None,
            model_abstained=False,
            attempts=attempts,
            error_type=type(last_error).__name__,
            error_message=str(last_error),
        )
