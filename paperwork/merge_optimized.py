#!/usr/bin/env python3
"""Merge optimized paper sections into one Markdown file."""

from __future__ import annotations

import argparse
from pathlib import Path


SECTION_ORDER = [
    "abstract optimized.md",
    "introduction optimized.md",
    "related work optimized.md",
    "motivation optimized.md",
    "problem formulation optimized.md",
    "algorithm design optimized.md",
    "performance analysis optimized.md",
    "conclusion optimized.md",
]


def strip_outer_fence(text: str) -> str:
    """Remove a single outer Markdown code fence when a section is fully fenced."""
    stripped = text.strip()
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def merge_sections(paperwork_dir: Path, keep_fences: bool) -> str:
    missing = [name for name in SECTION_ORDER if not (paperwork_dir / name).is_file()]
    if missing:
        missing_list = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(f"Missing optimized section files:\n{missing_list}")

    merged_parts = []
    for section_name in SECTION_ORDER:
        section_path = paperwork_dir / section_name
        content = section_path.read_text(encoding="utf-8")
        if not keep_fences:
            content = strip_outer_fence(content)
        merged_parts.append(content.rstrip())

    merged = "\n\n".join(merged_parts)
    if keep_fences:
        return merged + "\n"
    return f"```latex\n{merged}\n```\n"


def parse_args() -> argparse.Namespace:
    default_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Merge paperwork/* optimized.md files into a single Markdown file."
    )
    parser.add_argument(
        "-d",
        "--paperwork-dir",
        type=Path,
        default=default_dir,
        help="Directory containing the optimized section Markdown files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=default_dir / "optimized_full.md",
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--keep-fences",
        action="store_true",
        help="Keep each section file's own outer ```latex fence instead of wrapping the merged output once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paperwork_dir = args.paperwork_dir.resolve()
    output_path = args.output.resolve()

    merged = merge_sections(paperwork_dir, keep_fences=args.keep_fences)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(merged, encoding="utf-8")

    print(f"Merged {len(SECTION_ORDER)} sections into {output_path}")


if __name__ == "__main__":
    main()
