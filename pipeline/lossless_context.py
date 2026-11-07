"""Strict in-memory replay of one original MODIFY diff; no fuzzy patching."""

from __future__ import annotations

import re


class LosslessContextError(ValueError):
    pass


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$")


def original_modify_section(full_patch: str, path: str) -> str:
    """Extract an exact per-file section from the authoritative full patch."""
    header = f"diff --git a/{path} b/{path}"
    sections = re.split(r"(?=^diff --git )", full_patch, flags=re.MULTILINE)
    matches = [part for part in sections if part.splitlines()[0:1] == [header]]
    if len(matches) != 1:
        raise LosslessContextError("EXACT_ORIGINAL_FILE_SECTION_REQUIRED")
    return matches[0]


def replay_original_modify_diff(before: str, patch: str, path: str) -> str:
    """Return exact AFTER from complete BEFORE and a single-file original diff.

    Reject unsupported headers, overlapping hunks, offset/fuzzy application,
    count inconsistencies and any source-context mismatch. EOF markers preserve
    the absence of a final newline. This is deliberately a narrow text subset.
    """
    lines = patch.splitlines(keepends=True)
    if sum(line.startswith("diff --git ") for line in lines) != 1:
        raise LosslessContextError("SINGLE_FILE_ORIGINAL_DIFF_REQUIRED")
    if [line.rstrip("\r\n") for line in lines if line.startswith("--- ")] != [f"--- a/{path}"]:
        raise LosslessContextError("SOURCE_PATH_MISMATCH")
    if [line.rstrip("\r\n") for line in lines if line.startswith("+++ ")] != [f"+++ b/{path}"]:
        raise LosslessContextError("TARGET_PATH_MISMATCH")
    source = before.splitlines(keepends=True)
    output: list[str] = []
    cursor = 0
    index = 0
    hunks = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("@@ "):
            if hunks:
                raise LosslessContextError("UNSUPPORTED_DIFF_TRAILER")
            index += 1
            continue
        header = _HUNK.fullmatch(line.rstrip("\r\n"))
        if not header:
            raise LosslessContextError("INVALID_HUNK_HEADER")
        old_start, old_size, new_start, new_size = header.groups()
        old_count = int(old_size) if old_size is not None else 1
        new_count = int(new_size) if new_size is not None else 1
        old_offset = int(old_start) - (1 if old_count else 0)
        new_offset = int(new_start) - (1 if new_count else 0)
        if old_offset < cursor or old_offset > len(source):
            raise LosslessContextError("OVERLAPPING_OR_OUT_OF_RANGE_HUNK")
        output.extend(source[cursor:old_offset])
        if len(output) != new_offset:
            raise LosslessContextError("TARGET_HUNK_OFFSET_MISMATCH")
        index += 1
        body: list[tuple[str, str]] = []
        while index < len(lines) and not lines[index].startswith("@@ "):
            row = lines[index]
            if row.rstrip("\r\n") == "\\ No newline at end of file":
                if not body or not body[-1][1].endswith("\n"):
                    raise LosslessContextError("INVALID_EOF_MARKER")
                prefix, text = body[-1]
                body[-1] = (prefix, text[:-1])
            elif row[:1] in {" ", "-", "+"}:
                body.append((row[0], row[1:]))
            else:
                raise LosslessContextError("UNSUPPORTED_HUNK_BODY")
            index += 1
        old_lines = [text for prefix, text in body if prefix in {" ", "-"}]
        new_lines = [text for prefix, text in body if prefix in {" ", "+"}]
        if len(old_lines) != old_count or len(new_lines) != new_count:
            raise LosslessContextError("HUNK_LINE_COUNT_MISMATCH")
        if source[old_offset:old_offset + old_count] != old_lines:
            raise LosslessContextError("ORIGINAL_DIFF_SOURCE_CONTEXT_MISMATCH")
        output.extend(new_lines)
        cursor = old_offset + old_count
        hunks += 1
    if not hunks:
        raise LosslessContextError("ORIGINAL_DIFF_HUNKS_REQUIRED")
    output.extend(source[cursor:])
    return "".join(output)
