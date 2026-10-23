"""Tests: the command line must never show a traceback."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args: str) -> str:
    """Run `python -m src <args>` and return stdout plus stderr."""
    done = subprocess.run(
        [sys.executable, "-m", "src", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
        stdin=subprocess.DEVNULL,
    )
    return done.stdout + done.stderr


def assert_clean(output: str) -> None:
    """No Python traceback, and some message for the user."""
    assert "Traceback" not in output
    assert output.strip() != ""


@pytest.fixture
def indexed(tmp_path: Path) -> tuple[Path, Path]:
    """Index a tiny corpus through the CLI; return (raw, processed)."""
    raw = tmp_path / "raw"
    (raw / "docs").mkdir(parents=True)
    (raw / "docs" / "lora.md").write_text(
        "# LoRA adapters\n\nUse lora adapters to serve models.\n")
    proc = tmp_path / "proc"
    out = run_cli("index", "--max_chunk_size", "500",
                  "--raw_dir", str(raw), "--processed_dir", str(proc))
    assert "Traceback" not in out
    return raw, proc


def test_no_command_and_unknown_command() -> None:
    """Missing or unknown commands give usage, not a crash."""
    assert_clean(run_cli())
    assert_clean(run_cli("nonsense"))


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query(query: str, tmp_path: Path) -> None:
    """Empty queries are refused politely."""
    assert_clean(run_cli("search", query, "--k", "5",
                         "--processed_dir", str(tmp_path)))


@pytest.mark.parametrize("k", ["0", "-1", "abc", "true", "2.5"])
def test_bad_k(k: str, tmp_path: Path) -> None:
    """Invalid k values are refused politely."""
    assert_clean(run_cli("search", "hello", "--k", k,
                         "--processed_dir", str(tmp_path)))


def test_search_before_index(tmp_path: Path) -> None:
    """Searching without an index says so."""
    out = run_cli("search", "123", "--k", "3",
                  "--processed_dir", str(tmp_path / "nothing"))
    assert_clean(out)


@pytest.mark.parametrize("size", ["0", "-5", "5000", "abc"])
def test_index_bad_size(size: str, tmp_path: Path) -> None:
    """Chunk sizes outside 1..2000 are refused."""
    out = run_cli("index", "--max_chunk_size", size,
                  "--raw_dir", str(tmp_path),
                  "--processed_dir", str(tmp_path / "p"))
    assert_clean(out)
    assert not (tmp_path / "p").exists()


def test_index_missing_raw_dir(tmp_path: Path) -> None:
    """A missing corpus folder gives a clear message."""
    assert_clean(run_cli("index", "--raw_dir", str(tmp_path / "none"),
                         "--processed_dir", str(tmp_path / "p")))


def test_dataset_commands_bad_input(tmp_path: Path) -> None:
    """Missing options, files and malformed JSON never crash."""
    assert_clean(run_cli("search_dataset"))
    assert_clean(run_cli("search_dataset", "--dataset_path", "nope.json",
                         "--k", "5", "--save_directory", str(tmp_path)))
    broken = tmp_path / "broken.json"
    broken.write_text("{oops")
    assert_clean(run_cli("search_dataset", "--dataset_path", str(broken),
                         "--k", "5", "--save_directory", str(tmp_path)))
    assert_clean(run_cli("evaluate"))
    assert_clean(run_cli("evaluate",
                         "--student_search_results_path", str(broken),
                         "--dataset_path", str(broken)))


def test_evaluate_wrong_file_shape(tmp_path: Path) -> None:
    """A dataset passed as results gives a message, not a traceback."""
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"rag_questions": []}))
    assert_clean(run_cli("evaluate",
                         "--student_search_results_path", str(dataset),
                         "--dataset_path", str(dataset)))


def test_index_then_search(indexed: tuple[Path, Path]) -> None:
    """The happy path works through the real command line."""
    _, proc = indexed
    out = run_cli("search", "lora adapters", "--k", "3",
                  "--processed_dir", str(proc))
    assert "Traceback" not in out
    assert "lora.md" in out
