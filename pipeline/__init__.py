"""
VulnSage Pipeline Package
=========================

Hệ thống Kiểm chứng Bản vá Bảo mật Đa tệp bằng LLM
với kỹ thuật Prompting Phân rã Ngữ cảnh.

Modules:
    - llm_client: Module gọi API LLM dùng chung (retry, token accumulation)
    - data_parser: Tiền xử lý dữ liệu VulnSage G3
    - pipeline_p0: Baseline Think & Verify
    - pipeline_p1: Phân rã Cơ bản (CAHM-inspired)
    - pipeline_p2: Phân rã Nâng cao (SySeVR-inspired)
    - evaluator: Đánh giá và phân tích kết quả
"""
