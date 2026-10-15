import ast
import warnings
from src.models.models import Chunk
from .md_chunker import split_section, pack

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
