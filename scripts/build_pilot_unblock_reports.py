"""Build the offline-only pilot-unblock reports from immutable local artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MISMATCHES = {
    "16892": (27, 6, 23, 4, 0.424242), "17623": (6, 1, 6, 1, 0.571429),
    "7212": (4, 35, 3, 33, 0.923077), "16474": (11, 7, 4, 5, 0.5),
    "17164": (12, 8, 10, 6, 0.7), "13398": (24, 13, 7, 6, 0.243243),
    "16942": (36, 26, 23, 13, 0.516129), "18602": (22, 1, 20, 3, 0.913043),
    "5530": (35, 12, 21, 6, 0.574468), "16918": (23, 5, 23, 5, 0.714286),
    "18959": (26, 14, 7, 7, 0.3), "13216": (4, 2, 6, 3, 1.0),
    "16596": (56, 19, 58, 20, 0.893333), "17393": (12, 29, 8, 24, 0.243902),
    "10791": (64, 15, 13, 5, 0.139241), "17080": (61, 48, 46, 145, 0.816514),
    "18607": (82, 41, 80, 32, 0.845528), "8848": (10, 1, 2, 1, 0.272727),
}


def mismatch_category(values: tuple[int, int, int, int, float]) -> str:
    oa, od, ra, rd, _ = values
    if (ra + rd) < (oa + od):
        return "CSV_SNIPPET_OMITS_COMMIT_CHANGED_LINES"
    if (ra + rd) > (oa + od):
        return "SNIPPET_BOUNDARY_DIFFLIB_ADDS_DERIVED_CHANGES"
    if (ra, rd) != (oa, od):
        return "ADD_DELETE_DISTRIBUTION_DIFFERS"
    return "COUNTS_EQUAL_BUT_CHANGED_LINE_OR_HUNK_COVERAGE_DIFFERS"


def write_new(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--canonical-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    exact = json.loads(args.exact.read_text(encoding="utf-8"))
    canonical = json.loads(args.canonical_run.read_text(encoding="utf-8"))
    records = canonical["records"]
    policies = {item["policy_status"]: item for item in exact["policies"]}
    default_by_id = {item["source_id"]: item for item in policies["HISTORICAL_DEFAULT"]["samples"]}
    candidate_by_id = {item["source_id"]: item for item in policies["CANDIDATE_NEEDS_APPROVAL"]["samples"]}

    root_lines = [
        "# Root cause — 18 RECONSTRUCTED_CHANGE_COUNT_MISMATCH", "",
        "## Kết luận", "",
        "Cả 18 reason code đã được giải thích ở tầng engineering; **0 fixed, 18 explained, 0 unresolved**. Đây không phải mất dữ liệu trong canonical commit evidence. Validator so sánh số/dòng thay đổi của original commit patch với unified diff được dựng lại bằng `difflib` từ hai code snippet trong CSV. Hai snippet có thể là các lát mã độc lập, không phải toàn bộ file/hunk; vì vậy chúng có thể bỏ sót dòng của commit, tạo thay đổi giả ở biên snippet, hoặc có cùng tổng số nhưng không bao phủ đúng changed-line/hunk. Original patch và before/after blobs vẫn là nguồn chuẩn.", "",
        "Không sửa parser/normalizer và không tạo version canonical mới: sửa count để khớp CSV sẽ làm sai provenance. Waiver, nếu researcher chọn, chỉ được giới hạn ở `DERIVED_REPRESENTATION_ONLY`; không chứng minh security correctness.", "",
        "| source | original +/− | reconstructed +/− | line coverage | category |", "|---|---:|---:|---:|---|",
    ]
    for source_id, values in MISMATCHES.items():
        oa, od, ra, rd, ratio = values
        root_lines.append(f"| {source_id} | {oa}/{od} | {ra}/{rd} | {ratio:.6f} | {mismatch_category(values)} |")
    root_lines += [
        "", "## Điều kiện waiver không hợp lệ", "",
        "Không waiver nếu original patch/blob/hash/path/operation bị thiếu hoặc mâu thuẫn; nếu commit identity/parent không khớp; hoặc nếu gap nằm trong canonical evidence thay vì chỉ CSV-derived representation. Mỗi mẫu vẫn cần human decision riêng, không bulk-approve.", "",
        "## Nguồn kiểm tra", "", "- `dataset_validation/commit_verification.py`: logic đếm reconstructed lines và overlap với original hunks.",
        "- `data/pilot/canonical_pilot20_20261111_retry3/canonicalization_run.json`: counts, hashes và reason codes.",
        "- 18 original patches, retrieval manifests và blobs dưới `data/pilot/canonical_pilot20_20261111_retry3/evidence/`.", "",
    ]
    write_new(args.output / "change_count_root_cause.md", "\n".join(root_lines))

    security = """# Security review decision sheet

Trạng thái dưới đây là khuyến nghị kỹ thuật, **không phải researcher approval**. Ground truth của cả hai mẫu vẫn `PENDING`.

| source | CVE | exact commit / parent | security mechanism và vai trò file | derived gaps | proposal | gate |
|---|---|---|---|---|---|---|
| 18922 | CVE-2018-20961 | `7fafcfdf6377b18b2a726ea554d6e593ba44349f` / `4d8d5a392ae110d9b5889afd2b4beef9a09e712d` | `f_midi.c` tránh gọi `free_ep_req()` lần hai khi callback đã giải phóng buffer; `u_f.h` đặt `req->buf=NULL` sau `kfree` và cảnh báo double free. Hai file cùng thực thi invariant “buffer đã free không được free lại”. | None; commit verification `VERIFIED`, quality `PASS`. | **APPROVE candidate**, chỉ sau researcher kiểm tra source mapping, commit-unit completeness và ký decision có provenance. | HUMAN_DECISION_REQUIRED |
| 18867 | CVE-2017-17805 | `ecaaab5649781c5a0effdaf298a925063020500e` / `af3ff8045bbf3e32f1a448542e73abb4c8ceb6f1` | Cả generic và x86 Salsa20 bỏ fast path gọi `blkcipher_walk_done()` khi zero-length để API chỉ được gọi khi `walk.nbytes>0`; sửa đồng nhất cả hai implementation. | None; commit verification `VERIFIED`, quality `PASS`. | **APPROVE candidate**, chỉ sau researcher kiểm tra source mapping, commit-unit completeness và ký decision có provenance. | HUMAN_DECISION_REQUIRED |

## Bằng chứng độc lập đã truy cập

- 18922: [Ubuntu CVE-2018-20961](https://ubuntu.com/security/CVE-2018-20961) ghi “introduced by ad0d1a0, fixed by 7fafcfd”; [Debian tracker](https://security-tracker.debian.org/tracker/CVE-2018-20961) nêu exact upstream fix `7fafcfdf...`.
- 18867: [NVD CVE-2017-17805](https://nvd.nist.gov/vuln/detail/CVE-2017-17805) mô tả zero-length Salsa20 ở cả generic và x86 và dẫn exact commit `ecaaab...`.
- Original commit JSON/patch/blobs của cả hai mẫu nằm trong canonical evidence directory; file lists, parent SHA và patch subject khớp.

## Checklist researcher

1. Xác nhận exact CVE↔commit mapping và commit là đơn vị remediation phù hợp.
2. Xác nhận cả file thay đổi đều security-relevant và không thiếu commit cùng family.
3. Chọn APPROVE/DEFER/REJECT, ghi reviewer identity, UTC timestamp, lý do và evidence refs.
4. Không đổi `PENDING` bằng script/bulk rule; chỉ package builder nhận signed review record hợp lệ.
"""
    write_new(args.output / "security_review_decision_sheet.md", security)

    matrix_path = args.output / "final_pilot_readiness_matrix.csv"
    if matrix_path.exists():
        raise FileExistsError(f"refusing to overwrite {matrix_path}")
    fields = ["source_id", "sample_id", "cve_id", "TECHNICALLY_VALID", "TOKEN_FIT", "SECURITY_REVIEWED", "GOLD_LABEL_VERIFIED", "CONTEXT_POLICY_APPROVED", "INFERENCE_AUTHORIZED", "OFFICIAL_READY", "recommended_staging", "blockers"]
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            sid = record["source_id"]
            default = default_by_id[sid]
            candidate = candidate_by_id[sid]
            technically_valid = sid not in {"5530", "10791"}
            if default["all_static_prompts_fit"]:
                token_fit = "STATIC_PASS_RUNTIME_GUARD_REQUIRED"
                policy_gate = "HISTORICAL_DEFAULT"
            elif candidate["all_static_prompts_fit"]:
                token_fit = "CANDIDATE_STATIC_PASS_RUNTIME_GUARD_REQUIRED"
                policy_gate = "NEEDS_APPROVAL"
            elif candidate.get("preparation_error"):
                token_fit = "BLOCKED_CONTEXT_PREPARATION"
                policy_gate = "NEEDS_APPROVAL"
            else:
                token_fit = "STATIC_OVERFLOW"
                policy_gate = "NEEDS_APPROVAL"
            staged = sid in {"18922", "18867", "13398", "16918", "18607", "8848"}
            blockers = ["SECURITY_REVIEW_HUMAN_DECISION_REQUIRED", "GROUND_TRUTH_PENDING", "INFERENCE_NOT_AUTHORIZED"]
            if not technically_valid:
                blockers.append("OUT_OF_SCOPE" if sid == "5530" else "QUALIFICATION_BLOCKED")
            if token_fit != "STATIC_PASS_RUNTIME_GUARD_REQUIRED":
                blockers.append(token_fit)
            writer.writerow({
                "source_id": sid, "sample_id": record["sample_id"], "cve_id": record["cve_id"],
                "TECHNICALLY_VALID": str(technically_valid).upper(), "TOKEN_FIT": token_fit,
                "SECURITY_REVIEWED": "FALSE", "GOLD_LABEL_VERIFIED": "FALSE",
                "CONTEXT_POLICY_APPROVED": policy_gate, "INFERENCE_AUTHORIZED": "FALSE",
                "OFFICIAL_READY": "FALSE", "recommended_staging": str(staged).upper(),
                "blockers": "|".join(blockers),
            })

    decision = f"""# Pilot unblock decision

## Quyết định hiện tại: NO-GO cho official pilot

`OFFICIAL_READY = 0/20`. Không có quyền inference, không có gold-label researcher approval, và security review chưa được ký. Không có LLM/API/Ollama call trong nhiệm vụ này.

## Exact-token result

- Tokenizer assets local khớp 5/5 SHA-256; class `{exact['tokenizer']['class']}`, revision `{exact['tokenizer']['revision']}`.
- Runtime correspondence: empirical cho đúng model digest `{exact['model']['digest']}`; overhead quan sát 29 token được dành trước. Không khẳng định nguyên nhân overhead là chat template.
- Historical full-context: P0 6/20 static fit; P1 8/20; P2 8/20. Toàn bộ ba pipeline static fit ở 6/20.
- Candidate lossless delta: P0 10/20 static fit, 9 overflow, 1 context-preparation blocked; P1/P2 mỗi pipeline 12/20 fit, 7 overflow, 1 blocked. Toàn bộ ba pipeline static fit ở 10/20.
- P1/P2 synthesis là runtime-dependent và vẫn phải qua exact pre-call guard. Candidate policy chưa được researcher/GVHD phê duyệt.

## Deterministic staging list (không phải official cohort)

`18922, 18867, 13398, 16918, 18607, 8848`. Sáu mẫu này fit mọi static prompt dưới historical default. 18922 và 18867 có commit fidelity sạch và đã có decision sheet; bốn mẫu còn lại vẫn cần security review. Danh sách chỉ hỗ trợ ưu tiên review, không cấp quyền inference.

## Thay đổi engineering

Runner đã thêm `server_template_overhead_tokens` explicit, fail-closed với giá trị âm, đưa vào immutable run spec và trừ khỏi runtime exact input limit. Default 0 giữ tương thích; runtime tương ứng pinned smoke phải truyền 29 và kiểm tra lại model digest.

## Chủ thể cần xử lý blocker

- Researcher: ký security/ground-truth decisions riêng từng mẫu; quyết định waiver derived-only; cho phép inference đúng cohort.
- Researcher/GVHD: chấp thuận hoặc từ chối candidate context policy nếu muốn dùng thêm bốn mẫu static-fit; không đổi prompt học thuật trong báo cáo này.
- Engineering smoke owner: xác nhận Kaggle Ollama model digest và overhead trước run; runtime guard tự dừng nếu synthesis overflow.

## Điều kiện GO

Chỉ GO khi official package có review decisions hợp lệ, ordered list/hash bất biến, model digest/tokenizer/context policy đúng phê duyệt, exact static preflight PASS và quyền inference rõ ràng. Package/run command chỉ được tạo sau các gate đó.
"""
    write_new(args.output / "pilot_unblock_decision.md", decision)


if __name__ == "__main__":
    main()
