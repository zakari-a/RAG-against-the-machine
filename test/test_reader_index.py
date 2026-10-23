"""Tests: corpus reader, tokenizer, inverted index, save and load."""
import json
from pathlib import Path
from typing import Callable

import pytest

from src.corpus_read.corpus_reader import build_chunks, list_files, read_text
from src.indexer import InvertedIndex, build_index, load_all, save_all
from src.models import Chunk
from src.retriever import Retriever
from src.tokenizer import tokenize

MakeChunks = Callable[[list[str]], list[Chunk]]

# ---------------------------------------------------------- reader


def make_corpus(root: Path) -> None:
    """Create a small nested corpus with several file types."""
    (root / "sub").mkdir(parents=True)
    (root / "b.py").write_text("x = 1\n", encoding="utf-8")
    (root / "a.md").write_text("# A\ntext\n", encoding="utf-8")
    (root / "img.png").write_bytes(b"\x89PNG")
    (root / "sub" / "c.py").write_text("def f():\n    pass\n")
    (root / "sub" / "d.MD").write_text("# D\n", encoding="utf-8")


def test_list_files_filters_and_sorts(tmp_path: Path) -> None:
    """Only known suffixes, in a stable sorted order, at any depth."""
    make_corpus(tmp_path)
    files = list_files(str(tmp_path), {".py", ".md"})
    names = [p.relative_to(tmp_path).as_posix() for p in files]
    assert names == ["a.md", "b.py", "sub/c.py", "sub/d.MD"]
    assert files == list_files(str(tmp_path), {".py", ".md"})


def test_list_files_missing_folder(tmp_path: Path) -> None:
    """A missing corpus folder raises a clear error."""
    with pytest.raises(FileNotFoundError):
        list_files(str(tmp_path / "nope"), {".py"})


def test_read_text_keeps_exact_characters(tmp_path: Path) -> None:
    """CRLF and the BOM are kept, so offsets match the disk."""
    crlf = tmp_path / "crlf.md"
    crlf.write_bytes(b"a\r\nb\r\n")
    assert read_text(crlf) == "a\r\nb\r\n"
    bom = tmp_path / "bom.md"
    bom.write_bytes(b"\xef\xbb\xbfhi")
    assert read_text(bom) == "\ufeffhi"


def test_read_text_failures_return_none(tmp_path: Path) -> None:
    """Unreadable or missing files give None, never an exception."""
    bad = tmp_path / "bad.py"
    bad.write_bytes(b"\xff\xff")
    assert read_text(bad) is None
    assert read_text(tmp_path / "missing.py") is None


def test_build_chunks_paths_are_strings(tmp_path: Path) -> None:
    """Every chunk path is a str equal to the file's posix path."""
    make_corpus(tmp_path)
    chunks = build_chunks(str(tmp_path), 500)
    paths = {c.file_path for c in chunks}
    assert paths == {
        (tmp_path / "a.md").as_posix(),
        (tmp_path / "b.py").as_posix(),
        (tmp_path / "sub" / "c.py").as_posix(),
        (tmp_path / "sub" / "d.MD").as_posix(),
    }
    assert all(type(c.file_path) is str for c in chunks)


def test_build_chunks_matches_disk(tmp_path: Path) -> None:
    """Each chunk's text is the slice of the real file."""
    make_corpus(tmp_path)
    for chunk in build_chunks(str(tmp_path), 500):
        text = Path(chunk.file_path).read_text(encoding="utf-8")
        first, last = chunk.first_character_index, chunk.last_character_index
        assert text[first:last] == chunk.text


# -------------------------------------------------------- tokenizer


def test_tokenize_identifiers() -> None:
    """Identifiers are kept whole and split into parts."""
    assert tokenize("get_model_config") == [
        "get_model_config", "get", "model", "config"]
    assert tokenize("getModelConfig") == [
        "getmodelconfig", "get", "model", "config"]
    assert tokenize("HTTPServer") == ["httpserver", "http", "server"]
    assert tokenize("v2") == ["v2", "v", "2"]


def test_tokenize_plain_text_and_empty() -> None:
    """Plain words are lowercased, symbols are separators."""
    assert tokenize("How to configure the OpenAI server?") == [
        "how", "to", "configure", "the", "openai", "open", "ai", "server"]
    assert tokenize("") == []
    assert tokenize("?!  ---") == []
    assert tokenize("model") == ["model"]


# ------------------------------------------------------------ index


def test_build_index_example(four_chunks: list[Chunk]) -> None:
    """Postings and lengths match the hand-made example."""
    index = build_index(four_chunks)
    assert index.postings["model"] == [(0, 1), (1, 1), (2, 1)]
    assert index.postings["server"] == [(1, 2), (2, 1)]
    assert index.postings["config"] == [(0, 1), (1, 1)]
    assert index.postings["load"] == [(0, 1), (3, 1)]
    assert index.lengths == [3, 4, 3, 4]


def test_build_index_lengths_count_tokens(make_chunks: MakeChunks) -> None:
    """The length is the total token count, repeats included."""
    chunks = make_chunks(["a a a b", "!!!"])
    index = build_index(chunks)
    assert index.lengths == [4, 0]
    assert index.postings["a"] == [(0, 3)]


def test_build_index_empty() -> None:
    """No chunks give an empty index."""
    index = build_index([])
    assert index.postings == {} and index.lengths == []


# --------------------------------------------------- save and load


def test_save_load_round_trip(tmp_path: Path,
                              four_chunks: list[Chunk]) -> None:
    """Reloaded data is identical and gives identical search results."""
    index = build_index(four_chunks)
    save_all(four_chunks, index, str(tmp_path / "proc"), 2000)
    chunks, loaded, meta = load_all(str(tmp_path / "proc"))
    assert chunks == four_chunks
    assert loaded == index
    assert meta.num_chunks == 4 and meta.max_chunk_size == 2000
    before = Retriever(index, four_chunks).search("model server", 3)
    after = Retriever(loaded, chunks).search("model server", 3)
    assert before == after


def test_save_overwrites(tmp_path: Path, make_chunks: MakeChunks) -> None:
    """A second index run replaces the first."""
    folder = str(tmp_path / "proc")
    first = make_chunks(["one two"])
    save_all(first, build_index(first), folder, 100)
    second = make_chunks(["three"])
    save_all(second, build_index(second), folder, 100)
    assert load_all(folder)[0] == second


def test_load_missing_folder(tmp_path: Path) -> None:
    """Loading before indexing gives one clear error."""
    with pytest.raises(RuntimeError):
        load_all(str(tmp_path / "nothing"))


def test_load_corrupt_files(tmp_path: Path,
                            four_chunks: list[Chunk]) -> None:
    """A truncated or missing file is reported, not a crash."""
    folder = tmp_path / "proc"
    save_all(four_chunks, build_index(four_chunks), str(folder), 2000)
    (folder / "meta.json").unlink()
    with pytest.raises(RuntimeError):
        load_all(str(folder))
    save_all(four_chunks, build_index(four_chunks), str(folder), 2000)
    text = (folder / "index.json").read_text(encoding="utf-8")
    (folder / "index.json").write_text(text[: len(text) // 2])
    with pytest.raises(RuntimeError):
        load_all(str(folder))
    (folder / "index.json").write_text(json.dumps({"x": 1}))
    with pytest.raises(RuntimeError):
        load_all(str(folder))


def test_index_model_json_keeps_tuples() -> None:
    """JSON turns pairs into lists, loading turns them back."""
    index = InvertedIndex(postings={"a": [(0, 2)]}, lengths=[2])
    again = InvertedIndex.model_validate_json(index.model_dump_json())
    assert again.postings["a"] == [(0, 2)]
