"""Tests on the real corpus. Run from the project root.

Skipped when data/raw (or the datasets) are not present.
"""
import ast
import warnings
from pathlib import Path

import pytest

from src.md_chunker import chunk_file
from src.corpus_reader import build_chunks, list_files, read_files
from src.evaluator import iou
from src.models import AnsweredQuestion, Chunk, MinimalSource, RagDataset

RAW = Path("data/raw")
DATASETS = sorted(Path("data/datasets").rglob("AnsweredQuestions/*.json"))
SIZES = [200, 1000, 2000]
needs_corpus = pytest.mark.skipif(
    not RAW.is_dir(), reason="data/raw is missing")


def problems(text: str, chunks: list[Chunk], size: int) -> list[str]:
    """List the broken properties of one file's chunks."""
    found: list[str] = []
    pos = 0
    for chunk in chunks:
        first, last = chunk.first_character_index, chunk.last_character_index
        if first != pos:
            found.append(f"gap or overlap at {pos}")
        if last - first > size:
            found.append(f"chunk too wide ({last - first})")
        if text[first:last] != chunk.text:
            found.append("text differs from slice")
        pos = last
    if pos != len(text):
        found.append("chunks do not cover the file")
    return found


@needs_corpus
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("suffix", [".md", ".py"])
def test_corpus_chunks_are_valid(suffix: str, size: int) -> None:
    """Every file tiles, fits the size and matches its slices."""
    failures: list[str] = []
    for path in list_files(str(RAW), {suffix}):
        text = read_files(path)
        if text is None:
            continue
        chunks = chunk_file(path.as_posix(), text, size)
        failures += [f"{path}: {p}" for p in problems(text, chunks, size)]
    assert not failures, "\n".join(failures[:20])


@needs_corpus
def test_python_files_mostly_parse() -> None:
    """Few files should need the plain-text fallback."""
    files = list_files(str(RAW), {".py"})
    failed = 0
    for path in files:
        text = read_files(path)
        if text is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            failed += 1
    assert files, "no python files found"
    assert failed / len(files) < 0.10, f"{failed} files do not parse"


@needs_corpus
def test_few_unreadable_files() -> None:
    """Almost every indexable file must be readable."""
    files = list_files(str(RAW), {".py", ".md"})
    unreadable = [p for p in files if read_files(p) is None]
    assert files, "no indexable files found"
    assert len(unreadable) / len(files) < 0.01, unreadable[:10]


@pytest.fixture(scope="module")
def corpus_chunks() -> list[Chunk]:
    """Chunk the whole corpus once for several tests."""
    if not RAW.is_dir():
        pytest.skip("data/raw is missing")
    return build_chunks(str(RAW), 2000)


@needs_corpus
def test_corpus_paths_and_widths(corpus_chunks: list[Chunk]) -> None:
    """Paths are strings with the corpus prefix, widths fit 2000."""
    prefix = RAW.as_posix() + "/"
    assert corpus_chunks, "no chunks produced"
    assert all(type(c.file_path) is str for c in corpus_chunks)
    assert all(c.file_path.startswith(prefix) for c in corpus_chunks)
    assert max(len(c.text) for c in corpus_chunks) <= 2000


def sources_of(dataset: RagDataset) -> list[MinimalSource]:
    """Every ground-truth source of an answered dataset."""
    return [s for q in dataset.rag_questions
            if isinstance(q, AnsweredQuestion) for s in q.sources]


@needs_corpus
@pytest.mark.parametrize("path", DATASETS, ids=str)
def test_ground_truth_is_reachable(path: Path,
                                   corpus_chunks: list[Chunk]) -> None:
    """Chunks must cover the correct sources, or recall has a ceiling."""
    dataset = RagDataset.model_validate_json(
        path.read_text(encoding="utf-8"))
    truth = sources_of(dataset)
    assert truth, "no sources parsed from the answered dataset"
    by_file: dict[str, list[Chunk]] = {}
    for chunk in corpus_chunks:
        by_file.setdefault(chunk.file_path, []).append(chunk)
    missing = [s for s in truth if s.file_path not in by_file]
    uncovered = [
        s for s in truth
        if not any(iou(c, s) >= 0.05 for c in by_file.get(s.file_path, []))
    ]
    note = "\n".join(f"{s.file_path} "
                     f"[{s.first_character_index}:{s.last_character_index}]"
                     for s in (missing or uncovered)[:10])
    assert len(missing) / len(truth) < 0.02, f"path mismatch:\n{note}"
    assert len(uncovered) / len(truth) < 0.05, f"not covered:\n{note}"
