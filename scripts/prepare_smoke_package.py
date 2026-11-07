"""CLI for the review-gated one-sample smoke package builder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support the documented direct invocation from the repository root without
# requiring installation or mutation of the caller's PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_validation.smoke_package import prepare_one_sample_smoke_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-row-id", required=True)
    parser.add_argument("--review-decision", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    result = prepare_one_sample_smoke_package(
        source_dataset_path=args.source_dataset,
        source_row_id=args.source_row_id,
        review_decision_path=args.review_decision,
        schema_path=args.schema,
        output_dir=args.output_dir,
        git_commit=args.git_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
