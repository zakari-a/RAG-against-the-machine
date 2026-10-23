"""Tests: models, Markdown chunking, Python chunking, router."""
import pytest

from src.chunkers.chunker import chunk_file
from src.chunkers.md_chunker import (chunk_markdown, cut_points,
                                     fence_ranges, heading_starts, pack,
                                     sections, split_piece, split_section)
from src.chunkers.py_chunker import (chunk_python, line_starts, safe_units,
                                     tile, unit_ranges)
from src.models import Chunk, MinimalSource

SIZES = [50, 200, 500, 2000]
BODY = "word " * 120
MD = (
    "intro\n\n# A\n\n" + (BODY + "\n\n") * 5
    + "```bash\n# comment\nls\n```\n\n## B\n\nshort\n# C\n"
    + "x" * 900 + "\n"
)
MD_TEXTS = {
    "normal": MD,
    "crlf": MD.replace("\n", "\r\n"),
    "empty": "",
    "no_headings": "no headings here",
    "only_heading": "# only heading",
    "long_line": "y" * 5000,
    "unclosed_fence": "# T\n```\n# not heading\n" + "z\n" * 400,
}
SRC = (
    "import os\nX = 1\n\n@dec\ndef f():\n    pass\n\nY = 2\n\n\n"
    "class C:\n    def m(self):\n        return 1\n"
)
BIG_CLASS = "class Big:\n" + "".join(
    f"    def m{i}(self):\n        return {i}\n\n" for i in range(60)
)
PY_TEXTS = {
    "normal": SRC,
    "crlf": SRC.replace("\n", "\r\n"),
    "empty": "",
    "comment_only": "# just a comment\n",
    "no_final_newline": "def f():\n    pass",
    "syntax_error": "def f(:\n    pass\n" * 100,
    "long_line": "x = '" + "y" * 5000 + "'\n",
    "async_def": "async def g():\n    return 1\n" * 30,
    "big_class": BIG_CLASS,
    "form_feed": "def a():\n    pass\n\x0c\ndef b():\n    pass\n",
}


def check_tiling(text: str, chunks: list[Chunk], size: int) -> None:
    """Chunks must tile the text, fit the size and match their slice."""
    pos = 0
    for chunk in chunks:
        first = chunk.first_character_index
        last = chunk.last_character_index
        assert first == pos
        assert 0 < last - first <= size
        assert text[first:last] == chunk.text
        pos = last
    assert pos == len(text)


# ----------------------------------------------------------- models


def test_chunk_rejects_bad_ranges() -> None:
    """Empty, inverted and negative ranges are refused."""
    for first, last in [(5, 5), (9, 3), (-1, 4)]:
        with pytest.raises(ValueError):
            Chunk(file_path="a", first_character_index=first,
                  last_character_index=last, text="x")


def test_chunk_to_source_drops_text() -> None:
    """A source has only the three grader fields."""
    chunk = Chunk(file_path="a.md", first_character_index=0,
                  last_character_index=3, text="abc")
    source = chunk.to_source()
    assert type(source) is MinimalSource
    assert "text" not in source.model_dump()


def test_chunk_json_round_trip() -> None:
    """A chunk survives JSON serialization unchanged."""
    chunk = Chunk(file_path="a.md", first_character_index=2,
                  last_character_index=9, text="abcdefg")
    again = Chunk.model_validate_json(chunk.model_dump_json())
    assert again == chunk


# --------------------------------------------------------- markdown


def test_fence_ranges_pairs_and_slices() -> None:
    """A closed fence gives one range from opener to closer."""
    text = "a\n```bash\n# c\n```\nb\n"
    ranges = fence_ranges(text)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert text[start:end] == "```bash\n# c\n```"


def test_fence_unclosed_runs_to_end() -> None:
    """An unclosed fence swallows the rest of the file."""
    text = "x\n```\ncode\n"
    assert fence_ranges(text) == [(2, len(text))]


def test_fence_closing_rules() -> None:
    """A closer needs the same character, enough length, no info text."""
    assert fence_ranges("```\n~~~\n```\n") == [(0, 11)]
    assert fence_ranges("````\n```\n````\n") == [(0, 13)]
    text = "```\n```python\ncode\n"
    assert fence_ranges(text) == [(0, len(text))]


def test_heading_starts_ignore_fences_and_fakes() -> None:
    """Only real headings outside fences count."""
    text = (
        "# A\ntext\n```bash\n# c\n```\n## B\n#hashtag\n"
        "####### seven\n    # indented code\n"
    )
    assert heading_starts(text, fence_ranges(text)) == [
        0, text.index("## B")]


def test_sections_tile_the_text() -> None:
    """Sections cover the text with no gap and no duplicate start."""
    assert sections("", []) == []
    assert sections("abc", []) == [(0, 3)]
    text = "# A\nx\n# B\ny\n"
    assert sections(text, [0, 6]) == [(0, 6), (6, 12)]
    assert sections("pre\n# A\n", [4]) == [(0, 4), (4, 8)]


def test_cut_points_after_blank_lines() -> None:
    """Cuts land just after each blank line, LF or CRLF."""
    text = "aaa\n\nbbb\n\nccc\n"
    assert cut_points(text, 0, len(text)) == [5, 10]
    crlf = "aaa\r\n\r\nbbb\r\n"
    assert cut_points(crlf, 0, len(crlf)) == [7]


def test_split_piece_respects_size() -> None:
    """Pieces never exceed the size and cut at line breaks if possible."""
    flat = "a" * 10
    parts = split_piece(flat, 0, 10, 3)
    assert parts[0][0] == 0 and parts[-1][1] == 10
    assert all(0 < b - a <= 3 for a, b in parts)
    lines = "line\n" * 10
    for a, b in split_piece(lines, 0, len(lines), 12):
        assert b - a <= 12 and lines[a:b].endswith("\n")


def test_pack_merges_small_pieces() -> None:
    """Consecutive small pieces are merged while they fit."""
    assert pack([(0, 5), (5, 10), (10, 30)], 12) == [(0, 10), (10, 30)]
    assert pack([(0, 2), (2, 4), (4, 6)], 100) == [(0, 6)]
    assert pack([], 10) == []


def test_split_section_boundary() -> None:
    """A section of exactly the size stays whole, one more splits."""
    text = "x" * 10
    assert split_section(text, 0, 10, 10) == [(0, 10)]
    parts = split_section(text, 0, 10, 9)
    assert len(parts) == 2 and all(b - a <= 9 for a, b in parts)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", sorted(MD_TEXTS))
def test_markdown_properties(name: str, size: int) -> None:
    """Every sample tiles, fits the size and matches its slices."""
    text = MD_TEXTS[name]
    check_tiling(text, chunk_markdown("f.md", text, size), size)


def test_markdown_comment_is_not_a_boundary() -> None:
    """The bash comment inside the fence never starts a chunk."""
    chunks = chunk_markdown("f.md", MD, 2000)
    assert not any(c.text.startswith("# comment") for c in chunks)


def test_markdown_rejects_bad_size() -> None:
    """A non-positive size is refused."""
    for size in (0, -5):
        with pytest.raises(ValueError):
            chunk_markdown("f.md", "# a\n", size)


def test_markdown_keeps_file_path() -> None:
    """The path string is stored untouched."""
    chunks = chunk_markdown("data/raw/x/y.md", "# a\ntext\n", 100)
    assert {c.file_path for c in chunks} == {"data/raw/x/y.md"}


# ----------------------------------------------------------- python


def test_line_starts() -> None:
    """Offsets of each line start, counting every character."""
    assert line_starts("a\nbb\n\nc") == [0, 2, 5, 6]
    assert line_starts("") == [0]
    assert line_starts("a\r\nb") == [0, 3]


def test_unit_ranges_include_decorators() -> None:
    """A decorated function starts at its first decorator."""
    ranges = unit_ranges(SRC, line_starts(SRC))
    func_start = SRC.index("@dec")
    func_end = SRC.index("    pass\n") + len("    pass\n")
    assert ranges == [
        (func_start, func_end), (SRC.index("class C"), len(SRC))]


def test_tile_covers_everything() -> None:
    """Gap code attaches to the unit before it."""
    units = unit_ranges(SRC, line_starts(SRC))
    func_start, class_start = units[0][0], units[1][0]
    assert tile(SRC, units) == [
        (0, func_start), (func_start, class_start),
        (class_start, len(SRC))]
    assert tile("import os\n", []) == [(0, 10)]
    assert tile("", []) == []


def test_safe_units_on_syntax_error() -> None:
    """A file that does not parse gives no units, not an exception."""
    bad = "def f(:\n    pass\n"
    assert safe_units(bad, line_starts(bad)) == []


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("name", sorted(PY_TEXTS))
def test_python_properties(name: str, size: int) -> None:
    """Every sample tiles, fits the size and matches its slices."""
    text = PY_TEXTS[name]
    check_tiling(text, chunk_python("f.py", text, size), size)


@pytest.mark.parametrize("name", ["normal", "async_def", "form_feed"])
def test_python_units_start_on_definitions(name: str) -> None:
    """Each unit starts on a def, class, async def or decorator."""
    text = PY_TEXTS[name]
    for start, _ in unit_ranges(text, line_starts(text)):
        assert text[start:].startswith(("def ", "class ", "@", "async "))


def test_python_big_class_is_split() -> None:
    """A class wider than the size becomes several chunks."""
    chunks = chunk_python("f.py", BIG_CLASS, 300)
    assert len(chunks) > 3


# ----------------------------------------------------------- router


def test_router_dispatch_and_path() -> None:
    """Extensions pick the strategy and the path stays untouched."""
    py = chunk_file("a/b.py", "x = 1\n", 100)
    md = chunk_file("a/b.md", "# T\n", 100)
    assert py[0].file_path == "a/b.py"
    assert md[0].file_path == "a/b.md"


def test_router_unknown_and_case() -> None:
    """Unknown or missing extensions give nothing, case is ignored."""
    assert chunk_file("a/b.png", "abc", 100) == []
    assert chunk_file("a/Makefile", "all:\n", 100) == []
    assert chunk_file("a/b.pyi", "x: int\n", 100) == []
    assert chunk_file("a/B.PY", "x = 1\n", 100) != []
