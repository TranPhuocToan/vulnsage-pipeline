"""
pipeline/evaluator.py – MODULE 5: Đánh giá và Phân tích.

Chức năng:
    - Tính toán metrics: Accuracy, Precision, Recall, F1, Not Sure Rate
    - Chạy Ablation Study: P0 vs P1 vs P2
    - Tính Evidence Precision từ Manual Verification
    - Tạo biểu đồ so sánh

Quy tắc Invariant I7:
    NOT_SURE KHÔNG được gộp vào Accuracy.
    Chỉ tính Accuracy trên các mẫu có prediction ∈ {PATCHED, NOT_PATCHED}.
    Not Sure Rate được báo cáo như một chỉ số RIÊNG.

Metrics theo Research Questions:
    | Metric              | RQ  | Mô tả                                    |
    |---------------------|-----|-------------------------------------------|
    | Accuracy            | RQ1 | (TP+TN)/(TP+TN+FP+FN), loại NOT_SURE     |
    | Precision           | RQ1 | TP/(TP+FP)                                |
    | Recall              | RQ1 | TP/(TP+FN)                                |
    | F1-score            | RQ1 | Harmonic mean P & R                       |
    | Not Sure Rate       | RQ2 | NOT_SURE/total_samples                    |
    | Evidence Precision  | RQ2 | SUPPORTED/(SUPPORTED+UNSUPPORTED)         |
    | Avg Input Tokens    | RQ3 | Trung bình input tokens/mẫu              |
    | Avg Output Tokens   | RQ3 | Trung bình output tokens/mẫu             |
    | Avg Total Tokens    | RQ3 | Input + Output                            |
    | Avg Inference Time  | RQ3 | Giây/mẫu                                  |

Input:  results/main/p0_results.json, p1_results.json, p2_results.json
Output: results/main/evaluation_report.json

References:
    - Phần 9 – PIPELINE_ROADMAP.md
    - Invariant I7: NOT_SURE không gộp vào Accuracy
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Thêm project root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_MAIN_DIR, RESULTS_PILOT_DIR

# Cấu hình matplotlib cho non-interactive backend
matplotlib.use("Agg")


class PositiveOnlyEvaluationError(ValueError):
    """Raised when a positive-only result set is internally inconsistent."""


def evaluate_positive_only(
    results: List[Dict[str, Any]], expected_sample_ids: List[str]
) -> Dict[str, Any]:
    """Evaluate known fix-commit post-states without claiming binary accuracy.

    C=PATCHED, R=NOT_PATCHED, U=model NOT_SURE, E=technical/missing.
    Coverage and selective accuracy exclude abstentions and technical failures.
    """
    if len(expected_sample_ids) != len(set(expected_sample_ids)):
        raise PositiveOnlyEvaluationError("duplicate sample ID in expected list")
    by_id: Dict[str, Dict[str, Any]] = {}
    for result in results:
        sample_id = result.get("sample_id")
        if sample_id in by_id:
            raise PositiveOnlyEvaluationError(f"duplicate result sample ID: {sample_id}")
        if sample_id not in expected_sample_ids:
            raise PositiveOnlyEvaluationError(f"unexpected result sample ID: {sample_id}")
        by_id[sample_id] = result

    counts = {"C": 0, "R": 0, "U": 0, "E": 0}
    for sample_id in expected_sample_ids:
        result = by_id.get(sample_id)
        if not result or result.get("execution_status") != "COMPLETED":
            counts["E"] += 1
            continue
        prediction = result.get("prediction")
        if prediction == "PATCHED":
            counts["C"] += 1
        elif prediction == "NOT_PATCHED":
            counts["R"] += 1
        elif prediction == "NOT_SURE":
            counts["U"] += 1
        else:
            counts["E"] += 1

    n = len(expected_sample_ids)
    if sum(counts.values()) != n:
        raise PositiveOnlyEvaluationError("invariant violation: C + R + U + E != N")
    decided = counts["C"] + counts["R"]

    def rate(value: int, denominator: int) -> Optional[float]:
        return round(value / denominator, 6) if denominator else None

    return {
        "evaluation_scope": "POSITIVE_ONLY_KNOWN_FIX_COMMIT_POST_STATES",
        "N": n,
        **counts,
        "positive_acceptance_rate": rate(counts["C"], n),
        "false_rejection_rate": rate(counts["R"], n),
        "not_sure_rate": rate(counts["U"], n),
        "technical_failure_rate": rate(counts["E"], n),
        "coverage": rate(decided, n),
        "selective_accuracy": rate(counts["C"], decided),
        "binary_accuracy_f1_reported": False,
    }


def evaluate_positive_only_ablation(
    results_by_pipeline: Dict[str, List[Dict[str, Any]]],
    expected_sample_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Require P0/P1/P2 to use exactly the same immutable sample list."""
    required = {"P0", "P1", "P2"}
    if set(results_by_pipeline) != required:
        raise PositiveOnlyEvaluationError("results must contain exactly P0, P1, and P2")
    expected_set = set(expected_sample_ids)
    for pipeline, results in results_by_pipeline.items():
        ids = [item.get("sample_id") for item in results]
        if len(ids) != len(set(ids)):
            raise PositiveOnlyEvaluationError(f"{pipeline}: duplicate sample ID")
        if set(ids) != expected_set:
            raise PositiveOnlyEvaluationError(f"{pipeline}: sample list mismatch")
    return {
        pipeline: evaluate_positive_only(results, expected_sample_ids)
        for pipeline, results in results_by_pipeline.items()
    }


def evaluate_pipeline(
    results_path: str, pipeline_name: str
) -> Dict[str, Any]:
    """
    Đánh giá một pipeline dựa trên kết quả.

    ⚠ Invariant I7: NOT_SURE KHÔNG được gộp vào Accuracy.
    Accuracy chỉ tính trên mẫu có prediction ∈ {PATCHED, NOT_PATCHED}.

    Args:
        results_path: Đường dẫn đến file results JSON.
        pipeline_name: Tên pipeline (vd: "P0 (Baseline Think & Verify)").

    Returns:
        dict: Dictionary chứa tất cả metrics (Accuracy, P, R, F1,
              Not Sure Rate, Token stats, Timing).
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results: List[Dict] = json.load(f)

    if not results:
        return {"pipeline": pipeline_name, "error": "No results found"}

    # Tách NOT_SURE ra (Invariant I7)
    not_sure: List[Dict] = [
        r for r in results if r.get("prediction") == "NOT_SURE"
    ]
    sure_results: List[Dict] = [
        r for r in results if r.get("prediction") != "NOT_SURE"
    ]

    # Not Sure Rate (RQ2)
    not_sure_rate: float = len(not_sure) / len(results) if results else 0.0

    # Classification metrics (RQ1) – chỉ trên sure_results
    metrics: Dict[str, Any] = {
        "pipeline": pipeline_name,
        "total_samples": len(results),
        "evaluated_samples": len(sure_results),
        "not_sure_count": len(not_sure),
        "not_sure_rate": round(not_sure_rate, 4),
    }

    if sure_results:
        # Binary: PATCHED = 1, NOT_PATCHED = 0
        y_true: List[int] = [
            1 if r["ground_truth"] == "PATCHED" else 0
            for r in sure_results
        ]
        y_pred: List[int] = [
            1 if r["prediction"] == "PATCHED" else 0
            for r in sure_results
        ]

        metrics.update({
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(
                precision_score(y_true, y_pred, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, y_pred, zero_division=0), 4
            ),
            "f1_score": round(
                f1_score(y_true, y_pred, zero_division=0), 4
            ),
        })

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = {
            "TN": int(cm[0][0]),
            "FP": int(cm[0][1]),
            "FN": int(cm[1][0]),
            "TP": int(cm[1][1]),
        }
    else:
        metrics.update({
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "confusion_matrix": {"TN": 0, "FP": 0, "FN": 0, "TP": 0},
        })

    # Token usage stats (RQ3)
    total_tokens_list: List[int] = [
        r.get("token_usage", {}).get("total_tokens", 0) for r in results
    ]
    input_tokens_list: List[int] = [
        r.get("token_usage", {}).get("input_tokens", 0) for r in results
    ]
    output_tokens_list: List[int] = [
        r.get("token_usage", {}).get("output_tokens", 0) for r in results
    ]

    metrics.update({
        "avg_input_tokens": round(
            sum(input_tokens_list) / len(results), 0
        ),
        "avg_output_tokens": round(
            sum(output_tokens_list) / len(results), 0
        ),
        "avg_total_tokens": round(
            sum(total_tokens_list) / len(results), 0
        ),
        "total_tokens_all": sum(total_tokens_list),
    })

    # Timing stats (RQ3)
    inference_times: List[float] = [
        r.get("inference_time_sec", 0.0) for r in results
    ]
    metrics.update({
        "avg_inference_time": round(
            sum(inference_times) / len(results), 2
        ),
        "total_inference_time": round(sum(inference_times), 2),
    })

    # LLM calls stats
    llm_calls: List[int] = [
        r.get("num_llm_calls", 1) for r in results
    ]
    metrics["avg_llm_calls"] = round(
        sum(llm_calls) / len(results), 1
    )

    # Prediction distribution
    pred_dist: Dict[str, int] = {}
    for r in results:
        pred = r.get("prediction", "UNKNOWN")
        pred_dist[pred] = pred_dist.get(pred, 0) + 1
    metrics["prediction_distribution"] = pred_dist

    # Ground truth distribution
    gt_dist: Dict[str, int] = {}
    for r in results:
        gt = r.get("ground_truth", "UNKNOWN")
        gt_dist[gt] = gt_dist.get(gt, 0) + 1
    metrics["ground_truth_distribution"] = gt_dist

    return metrics


def run_ablation_study(
    p0_path: str, p1_path: str, p2_path: str,
    output_dir: str = RESULTS_MAIN_DIR,
) -> pd.DataFrame:
    """
    Ablation Study: P0 vs P1 vs P2.

    Tạo bảng so sánh metrics giữa 3 pipeline và tính delta
    so với baseline P0.

    Args:
        p0_path: Đường dẫn results P0.
        p1_path: Đường dẫn results P1.
        p2_path: Đường dẫn results P2.
        output_dir: Thư mục output cho report và biểu đồ.

    Returns:
        pd.DataFrame: Bảng so sánh Ablation Study.
    """
    rows: List[Dict[str, Any]] = [
        evaluate_pipeline(p0_path, "P0 (Baseline Think & Verify)"),
        evaluate_pipeline(p1_path, "P1 (Decomp. Basic)"),
        evaluate_pipeline(p2_path, "P2 (Decomp. + Dep. Evidence)"),
    ]

    df = pd.DataFrame(rows)

    # Delta so với P0 (baseline)
    delta_metrics = ["accuracy", "f1_score", "not_sure_rate", "precision", "recall"]
    for metric in delta_metrics:
        if metric in df.columns:
            baseline_val = df[metric].iloc[0]
            df[f"{metric}_delta"] = df[metric] - baseline_val

    # Lưu report
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(output_dir, "ablation_study.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Tạo biểu đồ
    _create_ablation_charts(df, output_dir)

    print(f"[Evaluator] Ablation Study saved to:")
    print(f"  Report: {report_path}")
    print(f"  CSV: {csv_path}")
    print(f"\n{df.to_string()}")

    return df


def _create_ablation_charts(
    df: pd.DataFrame, output_dir: str
) -> None:
    """
    Tạo biểu đồ Ablation Study.

    Tạo 3 biểu đồ:
    1. Classification metrics (Accuracy, P, R, F1) – grouped bar chart
    2. Not Sure Rate – bar chart
    3. Token & Time costs – bar chart

    Args:
        df: DataFrame chứa ablation results.
        output_dir: Thư mục lưu biểu đồ.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    pipeline_labels = ["P0\n(Baseline)", "P1\n(Decomp.)", "P2\n(Decomp.+Dep.)"]

    # Chart 1: Classification Metrics
    ax1 = axes[0]
    chart_metrics = ["accuracy", "precision", "recall", "f1_score"]
    chart_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(pipeline_labels))
    width = 0.2

    for i, (metric, label) in enumerate(zip(chart_metrics, chart_labels)):
        if metric in df.columns:
            values = df[metric].tolist()
            ax1.bar(x + i * width, values, width, label=label)

    ax1.set_xlabel("Pipeline")
    ax1.set_ylabel("Score")
    ax1.set_title("Classification Metrics (RQ1)")
    ax1.set_xticks(x + 1.5 * width)
    ax1.set_xticklabels(pipeline_labels)
    ax1.legend(loc="lower right")
    ax1.set_ylim(0, 1.05)

    # Chart 2: Not Sure Rate
    ax2 = axes[1]
    if "not_sure_rate" in df.columns:
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        ax2.bar(pipeline_labels, df["not_sure_rate"].tolist(), color=colors)
        ax2.set_ylabel("Not Sure Rate")
        ax2.set_title("Not Sure Rate (RQ2)")
        ax2.set_ylim(0, max(0.5, max(df["not_sure_rate"]) * 1.2))

        # Annotate values
        for i, v in enumerate(df["not_sure_rate"].tolist()):
            ax2.text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")

    # Chart 3: Cost Metrics
    ax3 = axes[2]
    if "avg_total_tokens" in df.columns:
        ax3_twin = ax3.twinx()

        tokens = df["avg_total_tokens"].tolist()
        times = df.get("avg_inference_time", pd.Series([0, 0, 0])).tolist()

        bar_positions = np.arange(len(pipeline_labels))
        bars = ax3.bar(
            bar_positions - 0.15, tokens, 0.3,
            label="Avg Tokens", color="#FFD93D", alpha=0.8,
        )
        ax3_twin.bar(
            bar_positions + 0.15, times, 0.3,
            label="Avg Time (s)", color="#6BCB77", alpha=0.8,
        )

        ax3.set_xlabel("Pipeline")
        ax3.set_ylabel("Avg Total Tokens")
        ax3_twin.set_ylabel("Avg Inference Time (s)")
        ax3.set_title("Cost Metrics (RQ3)")
        ax3.set_xticks(bar_positions)
        ax3.set_xticklabels(pipeline_labels)
        ax3.legend(loc="upper left")
        ax3_twin.legend(loc="upper right")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "ablation_study.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluator] Chart saved: {chart_path}")


def evaluate_evidence_precision(
    manual_verification_path: str,
) -> Dict[str, Any]:
    """
    Tính Evidence Precision từ kết quả Manual Verification (RQ2).

    Evidence Precision = SUPPORTED / (SUPPORTED + UNSUPPORTED)

    Args:
        manual_verification_path: Đường dẫn đến manual_verification.json.

    Returns:
        dict: Evidence precision metrics.
    """
    with open(manual_verification_path, "r", encoding="utf-8") as f:
        verifications: List[Dict] = json.load(f)

    counts: Dict[str, int] = {
        "SUPPORTED": 0,
        "UNSUPPORTED": 0,
        "UNCERTAIN": 0,
    }

    for v in verifications:
        key = v.get("manual_label", "UNCERTAIN")
        counts[key] = counts.get(key, 0) + 1

    denom: int = counts["SUPPORTED"] + counts["UNSUPPORTED"]
    precision: float = counts["SUPPORTED"] / denom if denom > 0 else 0.0

    result = {
        "evidence_precision": round(precision, 4),
        **counts,
        "total_verified": sum(counts.values()),
    }

    print(f"[Evaluator] Evidence Precision: {precision:.2%}")
    print(f"  SUPPORTED: {counts['SUPPORTED']}, "
          f"UNSUPPORTED: {counts['UNSUPPORTED']}, "
          f"UNCERTAIN: {counts['UNCERTAIN']}")

    return result


def evaluate_per_cwe(
    results_path: str, pipeline_name: str
) -> pd.DataFrame:
    """
    Đánh giá metrics phân theo CWE type.

    Hữu ích để hiểu pipeline hoạt động tốt/kém trên loại lỗ hổng nào.

    Args:
        results_path: Đường dẫn results JSON.
        pipeline_name: Tên pipeline.

    Returns:
        pd.DataFrame: Bảng metrics per CWE.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        results: List[Dict] = json.load(f)

    # Nhóm theo CWE
    by_cwe: Dict[str, List[Dict]] = {}
    for r in results:
        cwe = r.get("cwe_id", "UNKNOWN")
        by_cwe.setdefault(cwe, []).append(r)

    rows: List[Dict[str, Any]] = []
    for cwe, cwe_results in sorted(by_cwe.items()):
        sure = [r for r in cwe_results if r.get("prediction") != "NOT_SURE"]
        not_sure = len(cwe_results) - len(sure)

        row: Dict[str, Any] = {
            "cwe_id": cwe,
            "total": len(cwe_results),
            "not_sure": not_sure,
            "not_sure_rate": round(not_sure / len(cwe_results), 4) if cwe_results else 0,
        }

        if sure:
            y_true = [1 if r["ground_truth"] == "PATCHED" else 0 for r in sure]
            y_pred = [1 if r["prediction"] == "PATCHED" else 0 for r in sure]
            row["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            row["f1_score"] = round(
                f1_score(y_true, y_pred, zero_division=0), 4
            )
        else:
            row["accuracy"] = 0.0
            row["f1_score"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(metrics: Dict[str, Any]) -> None:
    """
    In tóm tắt metrics ra console.

    Args:
        metrics: Dictionary metrics từ evaluate_pipeline().
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT: {metrics['pipeline']}")
    print(f"{'='*60}")
    print(f"  Total Samples:    {metrics['total_samples']}")
    print(f"  Evaluated:        {metrics['evaluated_samples']}")
    print(f"  Not Sure:         {metrics['not_sure_count']} "
          f"({metrics['not_sure_rate']:.1%})")
    print(f"  ---")
    print(f"  Accuracy:         {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision:        {metrics.get('precision', 0):.4f}")
    print(f"  Recall:           {metrics.get('recall', 0):.4f}")
    print(f"  F1-score:         {metrics.get('f1_score', 0):.4f}")
    print(f"  ---")
    print(f"  Avg Tokens:       {metrics.get('avg_total_tokens', 0):.0f}")
    print(f"  Avg Time:         {metrics.get('avg_inference_time', 0):.2f}s")
    print(f"  Avg LLM Calls:    {metrics.get('avg_llm_calls', 0):.1f}")
    print(f"{'='*60}\n")


def main() -> None:
    """
    Entry point cho Evaluator module.

    Sử dụng:
        # Đánh giá một pipeline
        python -m pipeline.evaluator --mode single --input results/pilot/p0_pilot.json

        # Ablation Study (P0 vs P1 vs P2)
        python -m pipeline.evaluator --mode ablation \\
            --p0 results/main/p0_results.json \\
            --p1 results/main/p1_results.json \\
            --p2 results/main/p2_results.json

        # Pilot evaluation
        python -m pipeline.evaluator --mode pilot

        # Evidence Precision
        python -m pipeline.evaluator --mode evidence-precision \\
            --input results/main/manual_verification.json
    """
    parser = argparse.ArgumentParser(
        description="VulnSage Pipeline Evaluator"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "ablation", "pilot", "evidence-precision"],
        default="ablation",
        help="Chế độ đánh giá",
    )
    parser.add_argument("--input", type=str, help="Input results file")
    parser.add_argument("--p0", type=str, help="P0 results path")
    parser.add_argument("--p1", type=str, help="P1 results path")
    parser.add_argument("--p2", type=str, help="P2 results path")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.input:
            print("[ERROR] --input required for single mode")
            sys.exit(1)
        metrics = evaluate_pipeline(args.input, "Single Pipeline")
        print_summary(metrics)

    elif args.mode == "pilot":
        print("[Evaluator] Running Pilot Evaluation...")
        pilot_dir = RESULTS_PILOT_DIR
        for pipeline_name, filename in [
            ("P0 (Baseline)", "p0_pilot.json"),
            ("P1 (Decomp.)", "p1_pilot.json"),
            ("P2 (Decomp.+Dep.)", "p2_pilot.json"),
        ]:
            path = os.path.join(pilot_dir, filename)
            if os.path.exists(path):
                metrics = evaluate_pipeline(path, pipeline_name)
                print_summary(metrics)
            else:
                print(f"[SKIP] {path} not found")

    elif args.mode == "ablation":
        p0 = args.p0 or os.path.join(RESULTS_MAIN_DIR, "p0_results.json")
        p1 = args.p1 or os.path.join(RESULTS_MAIN_DIR, "p1_results.json")
        p2 = args.p2 or os.path.join(RESULTS_MAIN_DIR, "p2_results.json")

        for path, name in [(p0, "P0"), (p1, "P1"), (p2, "P2")]:
            if not os.path.exists(path):
                print(f"[ERROR] {name} results not found: {path}")
                sys.exit(1)

        df = run_ablation_study(p0, p1, p2)

        # Print individual summaries
        for _, row in df.iterrows():
            metrics = row.to_dict()
            print_summary(metrics)

    elif args.mode == "evidence-precision":
        if not args.input:
            print("[ERROR] --input required for evidence-precision mode")
            sys.exit(1)
        result = evaluate_evidence_precision(args.input)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
