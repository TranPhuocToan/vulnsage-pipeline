"""Build a review-gated multi-sample pilot package without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_validation.pilot_package import prepare_multi_sample_pilot_package


def _decision_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for value in values:
        source_id, separator, path = value.partition("=")
        if not separator or not source_id or not path or source_id in mapping:
            raise argparse.ArgumentTypeError(
                "each --decision must be unique SOURCE_ROW_ID=PATH"
            )
        mapping[source_id] = Path(path)
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a non-overwriting canonical pilot package only after every "
            "sample has an attributable human review decision. No LLM is called."
        )
    )
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-row-id", action="append", required=True)
    parser.add_argument(
        "--decision", action="append", required=True,
        metavar="SOURCE_ROW_ID=PATH",
    )
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args(argv)
    decisions = _decision_mapping(args.decision)
    result = prepare_multi_sample_pilot_package(
        source_dataset_path=args.source_dataset,
        ordered_source_row_ids=args.source_row_id,
        review_decision_paths=decisions,
        schema_path=args.schema,
        output_dir=args.output,
        git_commit=args.git_commit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
