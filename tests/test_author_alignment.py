"""Regression tests for the author-alignment boundary of the KLTN P0."""

import json
import unittest

from pipeline.llm_client import StructuredLLMClient, assess_think_verify_payload


class ThinkVerifyQualityTests(unittest.TestCase):
    def test_substantive_fields_pass(self):
        result = assess_think_verify_payload({
            "think": "The changed guard prevents a second free of req->buf.",
            "verify": "Both modified files maintain the same ownership invariant.",
            "verdict": "PATCHED",
        })
        self.assertEqual(result, {"status": "SUBSTANTIVE", "reason_codes": []})

    def test_exact_prompt_examples_are_not_substantive(self):
        result = assess_think_verify_payload({
            "think": "Phân tích chi tiết...",
            "verify": "Quá trình kiểm chứng...",
            "verdict": "PATCHED",
        })
        self.assertEqual(result["status"], "NON_SUBSTANTIVE")
        self.assertEqual(result["reason_codes"], [
            "THINK_PLACEHOLDER_EXACT", "VERIFY_PLACEHOLDER_EXACT",
        ])

    def test_missing_and_empty_fields_are_distinguished(self):
        result = assess_think_verify_payload({"think": "  ", "verdict": "NOT_SURE"})
        self.assertEqual(result["reason_codes"], [
            "THINK_EMPTY", "VERIFY_MISSING_OR_NOT_STRING",
        ])

    def test_quality_check_does_not_change_verdict_or_raw_response(self):
        payload = {
            "think": "Phân tích chi tiết...",
            "verify": "Quá trình kiểm chứng...",
            "verdict": "PATCHED",
            "confidence": "HIGH",
            "reasoning": "fixture",
        }
        raw = json.dumps(payload, ensure_ascii=False)
        call = StructuredLLMClient(lambda prompt: (raw, None), max_attempts=1).invoke(
            "prompt", "verdict"
        )
        quality = assess_think_verify_payload(call.parsed)
        self.assertEqual(call.status, "OK")
        self.assertEqual(call.parsed["verdict"], "PATCHED")
        self.assertEqual(call.raw_response, raw)
        self.assertEqual(quality["status"], "NON_SUBSTANTIVE")


if __name__ == "__main__":
    unittest.main()
