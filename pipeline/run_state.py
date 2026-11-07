"""Immutable run specification and separate execution journal for Phase 3B."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from pipeline.canonical_adapter import safe_sample_key


class RunStateError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def hash_files(paths: Sequence[Path]) -> str:
    entries = []
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        entries.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return hash_value(entries)


def build_run_spec(
    *,
    run_id: str,
    dataset_sha256: str,
    sample_ids: Sequence[str],
    pipeline_id: str,
    pipeline_version: str,
    model_id: str,
    model_version: str,
    parameters: Mapping[str, Any],
    retry_policy: Mapping[str, Any],
    context_policy_sha256: str,
    prompt_hashes: Mapping[str, str],
    code_sha256: str,
    git_commit: str | None,
    artifact_hashes: Mapping[str, str] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    output_directory: str | None = None,
    resume_policy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not run_id or any(char in run_id for char in '<>:"/\\|?*'):
        raise RunStateError("run_id must be non-empty and Windows-safe")
    if len(sample_ids) != len(set(sample_ids)):
        raise RunStateError("duplicate sample_id in run specification")
    base = {
        "spec_version": "phase3b-run-spec-1.0",
        "run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "sample_ids": list(sample_ids),
        "sample_list_sha256": hash_value(list(sample_ids)),
        "pipeline_id": pipeline_id,
        "pipeline_version": pipeline_version,
        "model": {"id": model_id, "version": model_version},
        "parameters": dict(parameters),
        "retry_policy": dict(retry_policy),
        "context_policy_sha256": context_policy_sha256,
        "prompt_hashes": dict(prompt_hashes),
        "code_sha256": code_sha256,
        "git_commit": git_commit,
    }
    if artifact_hashes is not None:
        base["artifact_hashes"] = dict(artifact_hashes)
    if model_metadata is not None:
        base["model_metadata"] = dict(model_metadata)
    if output_directory is not None:
        base["output_directory"] = output_directory
    if resume_policy is not None:
        base["resume_policy"] = dict(resume_policy)
    return {**base, "spec_sha256": hash_value(base)}


def _atomic_create_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RunStateError(f"refusing to overwrite historical artifact: {path}")
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise RunStateError(f"refusing to overwrite historical artifact: {path}")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class RunWorkspace:
    """Run-scoped storage with immutable spec and append-only journal."""

    def __init__(self, root: Path, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)
        self.run_dir = Path(root) / self.spec["run_id"]
        self.spec_path = self.run_dir / "run_spec.json"
        self.journal_path = self.run_dir / "execution_journal.jsonl"

    def initialize(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.spec_path.exists():
            self.validate_resume()
        else:
            _atomic_create_json(self.spec_path, self.spec)

    def validate_resume(self) -> None:
        try:
            stored = json.loads(self.spec_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RunStateError(f"corrupt run specification: {exc}") from exc
        if stored != self.spec:
            raise RunStateError("RESUME_SPEC_MISMATCH")
        if stored.get("spec_sha256") != hash_value({k: v for k, v in stored.items() if k != "spec_sha256"}):
            raise RunStateError("stored run specification hash is inconsistent")
        if len(stored["sample_ids"]) != len(set(stored["sample_ids"])):
            raise RunStateError("duplicate sample_id in stored run specification")
        self._validate_journal()

    def _validate_journal(self) -> None:
        if not self.journal_path.exists():
            return
        for line_number, line in enumerate(self.journal_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                entry = json.loads(line)
            except Exception as exc:
                raise RunStateError(f"corrupt journal line {line_number}: {exc}") from exc
            if entry.get("spec_sha256") != self.spec["spec_sha256"]:
                raise RunStateError(f"journal/spec mismatch at line {line_number}")

    def append_event(self, event: Mapping[str, Any]) -> None:
        entry = {**dict(event), "spec_sha256": self.spec["spec_sha256"]}
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_result(self, result: Mapping[str, Any]) -> Path:
        sample_id = result.get("sample_id")
        if sample_id not in self.spec["sample_ids"]:
            raise RunStateError("result sample_id is outside the immutable sample list")
        if result.get("pipeline", "").lower() != self.spec["pipeline_id"].lower():
            raise RunStateError("result pipeline does not match run specification")
        payload = {**dict(result), "spec_sha256": self.spec["spec_sha256"]}
        path = self.run_dir / "results" / f"{safe_sample_key(sample_id)}.json"
        _atomic_create_json(path, payload)
        return path

    def load_results(self) -> List[Dict[str, Any]]:
        results_dir = self.run_dir / "results"
        if not results_dir.exists():
            return []
        results = []
        seen = set()
        for path in sorted(results_dir.glob("*.json")):
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RunStateError(f"corrupt result {path.name}: {exc}") from exc
            sample_id = result.get("sample_id")
            if sample_id in seen:
                raise RunStateError(f"duplicate result sample_id: {sample_id}")
            if sample_id not in self.spec["sample_ids"]:
                raise RunStateError(f"unexpected result sample_id: {sample_id}")
            if result.get("spec_sha256") != self.spec["spec_sha256"]:
                raise RunStateError(f"result/spec mismatch: {path.name}")
            seen.add(sample_id)
            results.append(result)
        return results
