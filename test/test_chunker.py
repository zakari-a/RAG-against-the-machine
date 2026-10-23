from pathlib import Path

import pytest

from src.md_chunker import python_ranges  # adjust to your module name

ROOT = Path("data/raw")
SIZES = [200, 500, 1000, 2000]


def read_text(path: Path) -> str:
    """Read a file exactly as stored on disk."""
    with open(path, "r", encoding="utf-8", newline="") as file:
        return file.read()


def problems(text: str, ranges: list[tuple[int, int]], size: int) -> list[str]:
    """List the broken properties for one file."""
    found = []
    if "".join(text[a:b] for a, b in ranges) != text:
        found.append("chunks do not rebuild the file")
    if any(not 0 < b - a <= size for a, b in ranges):
        found.append("empty or too wide chunk")
    return found


@pytest.mark.parametrize("size", SIZES)
def test_python_corpus(size: int) -> None:
    """Every .py file must be chunked correctly at this size."""
    files = sorted(ROOT.rglob("*.py"))
    if not files:
        pytest.skip("data/raw is empty")
    failures = []
    for path in files:
        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue
        for issue in problems(text, python_ranges(text, size), size):
            failures.append(f"{path}: {issue}")
    assert not failures, "\n".join(failures[:20])