"""
demo/app.py – Streamlit Demo Application.

Ứng dụng demo cho Đồ án Tốt nghiệp:
"Thiết kế Kỹ thuật Prompting Phân rã Ngữ cảnh
cho Kiểm chứng Bản vá Bảo mật Đa tệp bằng LLM"

Chức năng:
    - Cho phép paste mẫu G3 JSON và chạy phân tích
    - Lựa chọn Pipeline: P0 / P1 / P2
    - Hiển thị kết quả: verdict, tokens, thời gian
    - Xem chi tiết phân tích (local analyses, dependency evidence)

Chạy:
    streamlit run demo/app.py

References:
    - Phần 14 – PIPELINE_ROADMAP.md
"""

import json
import os
import sys
import time

import streamlit as st

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="VulnSage Decomposition Prompting Demo",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .verdict-patched {
        padding: 10px 20px;
        background-color: #d4edda;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        font-size: 1.2rem;
    }
    .verdict-not-patched {
        padding: 10px 20px;
        background-color: #f8d7da;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        font-size: 1.2rem;
    }
    .verdict-not-sure {
        padding: 10px 20px;
        background-color: #fff3cd;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        font-size: 1.2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.title("🔒 Kiểm chứng Bản vá Bảo mật – Decomposition Prompting")
st.caption("Demo Đồ án Tốt nghiệp | Trần Phước Toàn – 25550029 | UIT")

st.markdown("---")

# ===== SAMPLE DATA =====
SAMPLE_G3_JSON = json.dumps({
    "sample_id": "demo_001",
    "cve_id": "CVE-2023-XXXX",
    "cwe_id": "CWE-416",
    "label": "PATCHED",
    "noise_score": 0.1,
    "commit_hash": "abc123",
    "files": [
        {
            "file_name": "net/ipv4/tcp.c",
            "vulnerable_code": "void tcp_close(struct sock *sk) {\n    // ... code ...\n    // Missing free\n}",
            "patched_code": "void tcp_close(struct sock *sk) {\n    // ... code ...\n    kfree(sk);\n}",
            "git_diff": "--- a/net/ipv4/tcp.c\n+++ b/net/ipv4/tcp.c\n@@ -100,3 +100,4 @@\n void tcp_close(struct sock *sk) {\n     // ... code ...\n-    // Missing free\n+    kfree(sk);\n }",
            "changed_lines": [
                {"type": "removed", "content": "// Missing free"},
                {"type": "added", "content": "kfree(sk);"}
            ],
            "file_index": 0,
            "total_files": 2
        },
        {
            "file_name": "include/net/tcp.h",
            "vulnerable_code": "// Header file\nvoid tcp_close(struct sock *sk);",
            "patched_code": "// Header file\nvoid tcp_close(struct sock *sk);",
            "git_diff": "",
            "changed_lines": [],
            "file_index": 1,
            "total_files": 2
        }
    ],
    "full_git_diff": "--- a/net/ipv4/tcp.c\n+++ b/net/ipv4/tcp.c\n@@ -100,3 +100,4 @@\n-    // Missing free\n+    kfree(sk);",
    "metadata": {
        "project": "linux_kernel",
        "num_files": 2,
        "num_changed_lines": 2
    }
}, indent=2, ensure_ascii=False)

# ===== LAYOUT =====
col_cfg, col_main = st.columns([1, 3])

with col_cfg:
    st.subheader("⚙️ Cấu hình")

    pipeline_choice = st.selectbox(
        "Chọn Pipeline:",
        [
            "P0 – Baseline Think & Verify",
            "P1 – Decomposition Basic (CAHM)",
            "P2 – Decomp. + Dep. Evidence (SySeVR)",
        ],
    )

    st.markdown("---")

    st.markdown("**📝 Nhập mẫu G3 JSON:**")
    sample_input = st.text_area(
        "Paste sample JSON (G3 format):",
        value=SAMPLE_G3_JSON,
        height=400,
        label_visibility="collapsed",
    )

    run_btn = st.button("🚀 Chạy Phân tích", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("**ℹ️ Thông tin Pipeline:**")
    pipeline_info = {
        "P0": "Gom toàn bộ code → 1 lần gọi LLM (Baseline)",
        "P1": "Phân tích từng file → Tổng hợp (CAHM)",
        "P2": "Phân tích file + Truy vết phụ thuộc → Tổng hợp (SySeVR)",
    }
    for key, desc in pipeline_info.items():
        st.caption(f"**{key}:** {desc}")

with col_main:
    st.subheader("📊 Kết quả Phân tích")

    if run_btn and sample_input:
        try:
            sample = json.loads(sample_input)

            # Validate basic structure
            required_fields = ["sample_id", "files", "metadata"]
            missing = [f for f in required_fields if f not in sample]
            if missing:
                st.error(f"❌ Thiếu trường bắt buộc: {', '.join(missing)}")
                st.stop()

            with st.spinner(f"🔄 Đang phân tích bằng {pipeline_choice}..."):
                start = time.time()

                if "P0" in pipeline_choice:
                    from pipeline.pipeline_p0 import process_single_sample_p0
                    result = process_single_sample_p0(sample)
                elif "P1" in pipeline_choice:
                    from pipeline.pipeline_p1 import process_single_sample_p1
                    result = process_single_sample_p1(sample)
                else:
                    from pipeline.pipeline_p2 import process_single_sample_p2
                    result = process_single_sample_p2(sample)

            # ===== VERDICT DISPLAY =====
            verdict = result.get("prediction", "UNKNOWN")
            confidence = "N/A"
            try:
                from pipeline.llm_client import extract_json_from_response
                parsed_resp = json.loads(
                    extract_json_from_response(
                        result.get("raw_response", result.get("synthesis_response", "{}"))
                    )
                )
                confidence = parsed_resp.get("confidence", "N/A")
            except Exception:
                pass

            if verdict == "PATCHED":
                icon = "🟢"
                css_class = "verdict-patched"
                verdict_text = "✅ PATCHED – Bản vá có vẻ đã khắc phục lỗ hổng"
            elif verdict == "NOT_PATCHED":
                icon = "🔴"
                css_class = "verdict-not-patched"
                verdict_text = "❌ NOT PATCHED – Bản vá chưa khắc phục lỗ hổng"
            else:
                icon = "🟡"
                css_class = "verdict-not-sure"
                verdict_text = "⚠️ NOT SURE – Không đủ cơ sở để kết luận"

            st.markdown(
                f'<div class="{css_class}">{verdict_text}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("")

            # ===== METRICS =====
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🏷️ Kết quả", f"{icon} {verdict}")
            c2.metric("📊 Confidence", confidence)
            c3.metric("🪙 Tokens", f"{result['token_usage']['total_tokens']:,}")
            c4.metric("⏱️ Thời gian", f"{result['inference_time_sec']:.2f}s")

            st.markdown("---")

            # ===== SAMPLE INFO =====
            st.markdown("**📋 Thông tin mẫu:**")
            info_c1, info_c2, info_c3, info_c4 = st.columns(4)
            info_c1.markdown(f"**CVE:** {sample.get('cve_id', 'N/A')}")
            info_c2.markdown(f"**CWE:** {sample.get('cwe_id', 'N/A')}")
            info_c3.markdown(f"**Files:** {sample['metadata'].get('num_files', 0)}")
            info_c4.markdown(f"**LLM Calls:** {result.get('num_llm_calls', 1)}")

            st.markdown("---")

            # ===== TOKEN BREAKDOWN =====
            st.markdown("**🪙 Token Usage Breakdown:**")
            t1, t2, t3 = st.columns(3)
            t1.metric("Input Tokens", f"{result['token_usage']['input_tokens']:,}")
            t2.metric("Output Tokens", f"{result['token_usage']['output_tokens']:,}")
            t3.metric("Total Tokens", f"{result['token_usage']['total_tokens']:,}")

            # ===== DETAILED RESULTS =====
            st.markdown("---")

            # P1/P2: Show local analyses
            if "local_analyses" in result and result["local_analyses"]:
                with st.expander("📂 Phân tích cục bộ từng tệp", expanded=False):
                    for la in result["local_analyses"]:
                        st.markdown(f"**📄 {la['file_name']}** — "
                                    f"Nhận định: `{la.get('local_verdict', 'N/A')}`")
                        if la.get("key_observations"):
                            for obs in la["key_observations"]:
                                st.markdown(f"  - {obs}")
                        st.markdown("---")

            # P2: Show dependency evidence
            if "dependency_evidence" in result and result["dependency_evidence"]:
                dep = result["dependency_evidence"]
                with st.expander("🔗 Bằng chứng phụ thuộc liên tệp", expanded=False):
                    deps = dep.get("dependencies", [])
                    if deps:
                        st.markdown(
                            f"**Tìm thấy {len(deps)} phụ thuộc** | "
                            f"Coupling: `{dep.get('overall_coupling', 'N/A')}`"
                        )
                        for d in deps:
                            st.markdown(
                                f"- **[{d.get('source_file')}]** → "
                                f"**[{d.get('target_file')}]**: "
                                f"`{d.get('dependency_type')}` – "
                                f"{d.get('description', 'N/A')}"
                            )
                    else:
                        st.info("Không tìm thấy phụ thuộc liên tệp.")

            # Full result JSON
            with st.expander("📋 Xem chi tiết đầy đủ (JSON)", expanded=False):
                st.json(result)

        except json.JSONDecodeError as e:
            st.error(f"❌ JSON không hợp lệ: {e}")
        except ImportError as e:
            st.error(f"❌ Import error: {e}. Hãy đảm bảo đã cài đặt dependencies.")
        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            import traceback
            st.code(traceback.format_exc())

    elif not run_btn:
        st.info(
            "👈 Paste mẫu G3 JSON vào cột bên trái và nhấn **Chạy Phân tích** "
            "để bắt đầu."
        )

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🔒 VulnSage Decomposition Prompting | "
    "Đồ án Tốt nghiệp – UIT | "
    "Trần Phước Toàn – 25550029"
    "</div>",
    unsafe_allow_html=True,
)
