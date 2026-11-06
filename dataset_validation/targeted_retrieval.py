"""Retrieve only explicitly planned commit patches and file blobs.

No repository is cloned. Existing artifacts are never overwritten with different
bytes; reruns reuse byte-identical files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote


RETRIEVER_VERSION = "1.0.0"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_repo_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not value or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe repository path: {value!r}")
    return path.as_posix()


def create_retrieval_plan(
    source_id: str, repository: str, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    commit = str(evidence.get("sha") or "")
    parents = evidence.get("parents") or []
    parent = str(parents[0].get("sha") or "") if parents else ""
    if len(commit) != 40 or len(parent) != 40:
        raise ValueError("Evidence must contain full commit and parent SHA")
    resources = [
        {
            "kind": "ORIGINAL_DIFF",
            "side": None,
            "path": None,
            "commit_sha": commit,
            "url": f"https://github.com/{repository}/commit/{commit}.patch",
            "local_path": "original.patch",
            "required": True,
        }
    ]
    operations = []
    for item in evidence.get("files", []):
        status = str(item.get("status") or "").lower()
        current = _safe_repo_path(str(item.get("filename") or ""))
        previous = item.get("previous_filename")
        previous = _safe_repo_path(str(previous)) if previous else None
        before_path = previous if status == "renamed" else current
        after_path = current
        operations.append(
            {
                "status": status.upper().replace("REMOVED", "DELETED"),
                "path_before": None if status == "added" else before_path,
                "path_after": None if status == "removed" else after_path,
            }
        )
        if status != "added":
            resources.append(
                {
                    "kind": "BLOB",
                    "side": "BEFORE",
                    "path": before_path,
                    "commit_sha": parent,
                    "url": f"https://raw.githubusercontent.com/{repository}/{parent}/{quote(before_path, safe='/')}",
                    "local_path": f"blobs/before/{before_path}",
                    "required": True,
                }
            )
        if status != "removed":
            resources.append(
                {
                    "kind": "BLOB",
                    "side": "AFTER",
                    "path": after_path,
                    "commit_sha": commit,
                    "url": f"https://raw.githubusercontent.com/{repository}/{commit}/{quote(after_path, safe='/')}",
                    "local_path": f"blobs/after/{after_path}",
                    "required": True,
                }
            )
    return {
        "retriever_version": RETRIEVER_VERSION,
        "source_id": str(source_id),
        "repository": repository,
        "commit_sha": commit,
        "parent_commit_sha": parent,
        "operations": operations,
        "resources": resources,
    }


def _default_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "VulnSage-phase2e/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def retrieve_plan(
    plan: Mapping[str, Any],
    output_dir: Path,
    *,
    fetch: Callable[[str], bytes] = _default_fetch,
    max_resources: int = 100,
    max_total_bytes: int = 50 * 1024 * 1024,
) -> dict[str, Any]:
    resources = list(plan.get("resources", []))
    if len(resources) > max_resources:
        raise ValueError("Retrieval plan exceeds max_resources")
    records = []
    total_bytes = 0
    for resource in resources:
        relative = PurePosixPath(str(resource["local_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe local artifact path")
        target = output_dir.joinpath(*relative.parts)
        data = target.read_bytes() if target.exists() else fetch(str(resource["url"]))
        total_bytes += len(data)
        if total_bytes > max_total_bytes:
            raise ValueError("Retrieval plan exceeds max_total_bytes")
        digest = _sha256(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        records.append(
            {
                **dict(resource),
                "status": "VERIFIED",
                "sha256": digest,
                "size_bytes": len(data),
            }
        )
    manifest = {
        key: plan[key]
        for key in (
            "retriever_version",
            "source_id",
            "repository",
            "commit_sha",
            "parent_commit_sha",
            "operations",
        )
    }
    manifest["resources"] = records
    manifest["total_bytes"] = total_bytes
    manifest_path = output_dir / "retrieval_manifest.json"
    encoded = (json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if manifest_path.exists() and manifest_path.read_bytes() != encoded:
        raise FileExistsError("Refusing to overwrite a different retrieval manifest")
    if not manifest_path.exists():
        manifest_path.write_bytes(encoded)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-resources", type=int, default=100)
    parser.add_argument("--max-total-bytes", type=int, default=50 * 1024 * 1024)
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    plan = create_retrieval_plan(args.source_id, args.repository, evidence)
    start = time.perf_counter()
    manifest = retrieve_plan(
        plan,
        args.output_dir,
        max_resources=args.max_resources,
        max_total_bytes=args.max_total_bytes,
    )
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "resources": len(manifest["resources"]),
                "bytes": manifest["total_bytes"],
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
