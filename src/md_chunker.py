import re
from typing import Tuple
import ast
import warnings

from src.models import Chunk

Range = Tuple[int, int]

HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+|$).*?$", re.MULTILINE)
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})(.*)$", re.MULTILINE)
BLANK_LINE_RE = re.compile(r"^[ \t]*\r?$", re.MULTILINE)


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
            file_path: str, text: str,
            max_size: int = 2000) -> list[Chunk]:
    """Chunk one Markdown text; ``text`` must be exactly the file content."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    fences = fence_ranges(text)
    starts = heading_starts(text, fences)
    ranges: list[Range] = []
    for a, b in sections(text, starts):
        ranges.extend(split_section(text, a, b, max_size))
    return [
        Chunk(
            file_path=file_path,
            first_character_index=a,
            last_character_index=b,
            text=text[a:b]
        )
        for a, b in ranges
    ]

# python chunking part


DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def start_lines(node) -> int:
    lines = [node.lineno] + [d.lineno for d in node.decorator_list]
    return min(lines)


def tile(text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    starts = {a for a, _ in ranges}
    bounds = sorted({0, len(text)} | starts)
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if a < b]


def unit_ranges(text:str , starts: list[int]) -> list[tuple[int, int]]:
    ranges = []
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, DEFS):
            end_line = node.end_lineno or node.lineno
            first = starts[start_lines(node) - 1]
            last = starts[end_line] if end_line < len(starts) else len(text)
            ranges.append((first, last))
    return ranges


def line_starts(text: str) -> list[int]:
    """Return the offset where each line begins."""
    start = [0]
    for i, char in enumerate(text):
        if char == "\n":
            start.append(i + 1)
    return start


def safe_units(text: str, starts: list[tuple[int, int]]) -> list | list[tuple[int, int]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return unit_ranges(text, starts)
    except (SyntaxError, ValueError, RecursionError):
        return []


def python_ranges(text: str, size: int) -> list[tuple[int, int]]:
    starts = line_starts(text)
    units = tile(text, unit_ranges(text, starts))
    pieces = []
    for a, b in units:
        pieces.extend(split_section(text, a, b, size))
    return pack(pieces, size)

def chunk_python(file_path: str, text: str, size: int) -> list[Chunk]:
    ranges = python_ranges(text, size)
    return [
        Chunk(
            file_path=file_path,
            first_character_index=a,
            last_character_index=b,
            text=text[a:b]
        )
        for a, b in ranges
    ]

def chunk_text(file_path: str, text: str, size: int) -> list[Chunk]:
    """Chunk plain text by blank lines, then lines, then size."""
    if not text:
        return []
    ranges = split_section(text, 0, len(text), size)
    return [
        Chunk(file_path=file_path, first_character_index=a,
              last_character_index=b, text=text[a:b])
        for a, b in ranges
    ]


def chunk_file(file_path: str, text: str, size: int) -> list[Chunk]:
    """Pick the chunker from the extension, like the indexer does."""
    if file.path.endswith(".txt"):
        return chunk_text(file_path, text, size)
    if file_path.endswith(".py"):
        return chunk_python(file_path, text, size)
    if file_path.endswith(".md"):
        return chunk_markdown(file_path, text, size)
    return []