"""Markdown chunker built on regular expressions.

Pipeline: fenced code ranges -> real headings -> sections -> size-capped
chunks. Every offset is absolute (relative to the whole file) and the end
offset is exclusive, so ``text[first:last]`` is exactly the chunk.
"""
import re
import sys
from typing import Tuple

from pydantic import BaseModel, Field, model_validator

Range = Tuple[int, int]

HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$).*?$", re.MULTILINE)
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$", re.MULTILINE)
BLANK_LINE_RE = re.compile(r"^[ \t]*\r?$", re.MULTILINE)


class Chunk(BaseModel):
    """A chunk's location (and text) inside the original file."""

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int
    text: str

    @model_validator(mode="after")
    def check_order(self) -> "Chunk":
        """Reject empty or inverted ranges."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError("last index must be after first index")
        return self


def closes(opener: re.Match[str], candidate: re.Match[str]) -> bool:
    """Tell whether ``candidate`` is a valid closing fence for ``opener``."""
    open_fence, close_fence = opener.group(1), candidate.group(1)
    return (
        close_fence[0] == open_fence[0]
        and len(close_fence) >= len(open_fence)
        and not candidate.group(2).strip()
    )


def fence_ranges(text: str) -> list[Range]:
    """Return the (start, end) range of every fenced code block."""
    ranges: list[Range] = []
    opener: re.Match[str] | None = None
    for match in FENCE_RE.finditer(text):
        if opener is None:
            opener = match
        elif closes(opener, match):
            ranges.append((opener.start(), match.end()))
            opener = None
    if opener is not None:
        ranges.append((opener.start(), len(text)))
    return ranges


def inside(pos: int, ranges: list[Range]) -> bool:
    """Tell whether ``pos`` falls inside one of the ranges."""
    return any(start <= pos < end for start, end in ranges)


def heading_starts(text: str, fences: list[Range]) -> list[int]:
    """Return offsets of real headings (ignoring '#' inside fences)."""
    return [
        match.start()
        for match in HEADING_RE.finditer(text)
        if not inside(match.start(), fences)
    ]


def sections(text: str, starts: list[int]) -> list[Range]:
    """Turn heading offsets into ranges that tile the whole text."""
    bounds = sorted(set([0] + starts + [len(text)]))
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if a < b]


def cut_points(text: str, start: int, end: int) -> list[int]:
    """Return offsets just after each blank line inside [start, end)."""
    cuts: list[int] = []
    for match in BLANK_LINE_RE.finditer(text, start, end):
        newline = text.find("\n", match.end(), end)
        if newline != -1 and start < newline + 1 < end:
            cuts.append(newline + 1)
    return cuts


def split_piece(text: str, a: int, b: int, max_size: int) -> list[Range]:
    """Split one range at line breaks, then by size, into small parts."""
    parts: list[Range] = []
    while b - a > max_size:
        newline = text.rfind("\n", a, a + max_size)
        cut = newline + 1 if newline >= a else a + max_size
        parts.append((a, cut))
        a = cut
    parts.append((a, b))
    return parts


def pack(pieces: list[Range], max_size: int) -> list[Range]:
    """Greedily merge consecutive pieces while they still fit."""
    packed: list[Range] = []
    for a, b in pieces:
        if packed and b - packed[-1][0] <= max_size:
            packed[-1] = (packed[-1][0], b)
        else:
            packed.append((a, b))
    return packed


def split_section(text: str, start: int, end: int, size: int) -> list[Range]:
    """Return the chunk ranges of one section, each at most ``size``."""
    if end - start <= size:
        return [(start, end)]
    bounds = [start] + cut_points(text, start, end) + [end]
    pieces: list[Range] = []
    for a, b in zip(bounds, bounds[1:]):
        pieces.extend(split_piece(text, a, b, size))
    return pack(pieces, size)


def chunk_markdown(
    file_path: str, text: str, max_size: int = 2000
) -> list[Chunk]:
    """Chunk one Markdown text; ``text`` must be exactly the file content."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    fences = fence_ranges(text)
    starts = heading_starts(text, fences)
    chunks: list[Chunk] = []
    for a, b in sections(text, starts):
        for first, last in split_section(text, a, b, max_size):
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=first,
                    last_character_index=last,
                    text=text[first:last],
                )
            )
    return chunks


def main() -> None:
    """Demo: python md_chunker.py <file.md> [max_size]."""
    path = sys.argv[1]
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    with open(path, "r", encoding="utf-8", newline="") as file:
        text = file.read()
    for chunk in chunk_markdown(path, text, size):
        print(chunk.text)


if __name__ == "__main__":
    main()
