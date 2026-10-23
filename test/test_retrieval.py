"""Tests: BM25 retriever, evaluator, dataset loading, end to end."""
import json
import math
from pathlib import Path
from typing import Callable

import pytest

from src.evaluator import (evaluate_results, iou, is_found,
                           question_recall)
from src.indexer import build_index
from src.models import (AnsweredQuestion, Chunk, MinimalSearchResults,
                        MinimalSource, RagDataset, StudentSearchResults)
from src.pipeline import (load_dataset, make_retriever, run_index,
                          run_search_dataset)
from src.retriever import Retriever

MakeChunks = Callable[[list[str]], list[Chunk]]


def src(path: str, first: int, last: int) -> MinimalSource:
    """Build a MinimalSource quickly."""
    return MinimalSource(file_path=path, first_character_index=first,
                         last_character_index=last)


def approx(value: float, expected: float) -> bool:
    """Compare floats with a small tolerance."""
    return math.isclose(value, expected, abs_tol=2e-3)


# -------------------------------------------------------- retriever


def test_constructor_statistics(retriever: Retriever) -> None:
    """N and the average length come from the index."""
    assert retriever.n == 4
    assert retriever.avg_length == 3.5


def test_constructor_rejects_bad_input(four_chunks: list[Chunk]) -> None:
    """Mismatched lists and absurd settings fail early."""
    index = build_index(four_chunks)
    with pytest.raises(ValueError):
        Retriever(index, four_chunks[:3])
    with pytest.raises(ValueError):
        Retriever(index, four_chunks, b=2)
    with pytest.raises(ValueError):
        Retriever(index, four_chunks, k1=-1)


def test_empty_corpus() -> None:
    """An empty corpus builds and finds nothing."""
    empty = Retriever(build_index([]), [])
    assert empty.n == 0
    assert empty.search("anything", 5) == []


def test_idf_values(retriever: Retriever) -> None:
    """Rare terms weigh more, unknown terms weigh nothing."""
    assert approx(retriever.idf("model"), 0.3567)
    assert approx(retriever.idf("server"), 0.6931)
    assert retriever.idf("unknownword") == 0.0
    assert retriever.idf("server") > retriever.idf("model")


def test_term_score_values(retriever: Retriever) -> None:
    """Scores match the hand-computed table."""
    idf_model = retriever.idf("model")
    idf_server = retriever.idf("server")
    assert approx(retriever.term_score(idf_model, 1, 0), 0.381)
    assert approx(retriever.term_score(idf_model, 1, 1), 0.335)
    assert approx(retriever.term_score(idf_server, 2, 1), 0.947)


def test_term_score_saturates_and_normalizes(
        four_chunks: list[Chunk]) -> None:
    """More repeats add less each time; b=0 ignores chunk length."""
    index = build_index(four_chunks)
    plain = Retriever(index, four_chunks)
    scores = [plain.term_score(1.0, c, 0) for c in (1, 2, 5, 20)]
    assert scores == sorted(scores)
    gaps = [b - a for a, b in zip(scores, scores[1:])]
    assert gaps[0] > gaps[1] > gaps[2] or gaps[0] > gaps[2]
    assert scores[-1] < plain.k1 + 1
    flat = Retriever(index, four_chunks, b=0)
    assert flat.term_score(1.0, 1, 0) == flat.term_score(1.0, 1, 1)
    assert plain.term_score(1.0, 1, 0) != plain.term_score(1.0, 1, 1)


def test_score_whole_query(retriever: Retriever) -> None:
    """Only matching chunks are scored, with the expected values."""
    result = retriever.score("model server")
    assert set(result) == {0, 1, 2}
    assert approx(result[0], 0.381)
    assert approx(result[1], 1.282)
    assert approx(result[2], 1.122)


def test_score_edge_queries(retriever: Retriever) -> None:
    """Empty, symbol-only and unknown queries score nothing."""
    for query in ["", "???", "zzzunknown"]:
        assert retriever.score(query) == {}
    assert retriever.score("model model") == retriever.score("model")


def test_search_order_and_limits(retriever: Retriever) -> None:
    """Best first, k respected, no padding, no crash on bad k."""
    results = retriever.search("model server", 3)
    assert [r.file_path for r in results] == ["f1.md", "f2.md", "f0.md"]
    assert len(retriever.search("model server", 10)) == 3
    assert len(retriever.search("model server", 1)) == 1
    assert retriever.search("model server", 0) == []
    assert retriever.search("model server", -2) == []
    assert retriever.search("", 5) == []


def test_search_returns_plain_sources(retriever: Retriever) -> None:
    """Results are MinimalSource objects without text."""
    results = retriever.search("model server", 3)
    assert all(type(r) is MinimalSource for r in results)
    assert all("text" not in r.model_dump() for r in results)


def test_search_is_deterministic(retriever: Retriever) -> None:
    """The same query always gives the same ranking."""
    first = retriever.search("model server config", 4)
    assert all(retriever.search("model server config", 4) == first
               for _ in range(5))


def test_ties_break_by_chunk_id(make_chunks: MakeChunks) -> None:
    """Equal scores keep the lower chunk id first."""
    chunks = make_chunks(["alpha beta", "alpha beta", "gamma delta"])
    engine = Retriever(build_index(chunks), chunks)
    paths = [r.file_path for r in engine.search("alpha", 3)]
    assert paths == ["f0.md", "f1.md"]


def test_identifier_and_paraphrase(make_chunks: MakeChunks) -> None:
    """Quoting an identifier and paraphrasing it both find the chunk."""
    chunks = make_chunks([
        "def get_model_config(self): pass",
        "we get the model and the config here",
        "unrelated words only",
    ])
    engine = Retriever(build_index(chunks), chunks)
    assert engine.search("get_model_config", 1)[0].file_path == "f0.md"
    top2 = {r.file_path for r in engine.search("get the model config", 2)}
    assert top2 == {"f0.md", "f1.md"}


# -------------------------------------------------------- evaluator


def test_iou_values() -> None:
    """Overlap maths on simple ranges."""
    assert approx(iou(src("a", 0, 100), src("a", 50, 150)), 1 / 3)
    assert iou(src("a", 0, 100), src("a", 100, 200)) == 0.0
    assert iou(src("a", 0, 100), src("a", 0, 100)) == 1.0
    assert approx(iou(src("a", 0, 100), src("a", 0, 1000)), 0.1)


def test_is_found_rules() -> None:
    """Same file and enough overlap are both required."""
    truth = src("a.md", 0, 100)
    assert is_found(truth, [src("a.md", 0, 100)])
    assert not is_found(truth, [src("b.md", 0, 100)])
    assert not is_found(truth, [src("a.md", 95, 2095)])
    assert not is_found(truth, [])


def test_is_found_overlap_threshold() -> None:
    """A partial overlap counts from IoU 0.05 upwards."""
    truth = src("a.md", 0, 1000)
    assert is_found(truth, [src("a.md", 0, 100)])
    assert is_found(src("a.md", 0, 100), [src("a.md", 0, 5)])
    assert not is_found(truth, [src("a.md", 0, 40)])


def test_question_recall_and_k() -> None:
    """Recall is the found share and only the top k count."""
    truth = [src("a", 0, 10), src("b", 0, 10)]
    got = [src("x", 0, 10), src("a", 0, 10)]
    assert question_recall(truth, got, 1) == 0.0
    assert question_recall(truth, got, 2) == 0.5


def make_dataset(items: list[tuple[str, list[MinimalSource]]]) -> RagDataset:
    """Build an answered dataset from (id, sources) pairs."""
    return RagDataset(rag_questions=[
        AnsweredQuestion(question_id=qid, question=f"question {qid}?",
                         sources=sources, answer="x")
        for qid, sources in items
    ])


def make_results(items: list[tuple[str, list[MinimalSource]]],
                 k: int = 5) -> StudentSearchResults:
    """Build search results from (id, sources) pairs."""
    return StudentSearchResults(search_results=[
        MinimalSearchResults(question_id=qid, question=f"question {qid}?",
                             retrieved_sources=sources)
        for qid, sources in items
    ], k=k)


def test_perfect_run_scores_one() -> None:
    """Feeding the truth back gives full recall once k is large enough."""
    pairs = [("q1", [src("a", 0, 50)]),
             ("q2", [src("b", 10, 90), src("c", 0, 30)])]
    scores = evaluate_results(make_results(pairs), make_dataset(pairs),
                              [1, 3, 5])
    # q2 has two sources, so only one of them fits in the top 1.
    assert scores == {1: 0.75, 3: 1.0, 5: 1.0}


def test_empty_and_missing_results_score_zero() -> None:
    """Missing answers count as zero, an empty truth is not a crash."""
    truth = [("q1", [src("a", 0, 50)]), ("q2", [src("b", 0, 50)])]
    dataset = make_dataset(truth)
    empty = make_results([("q1", []), ("q2", [])])
    assert evaluate_results(empty, dataset, [5]) == {5: 0.0}
    partial = make_results([("q1", [src("a", 0, 50)])])
    assert evaluate_results(partial, dataset, [5]) == {5: 0.5}
    nothing = RagDataset(rag_questions=[])
    assert evaluate_results(partial, nothing, [5]) == {5: 0.0}


def test_missing_question_ids_are_random() -> None:
    """Without question_id each load invents new ids (a pitfall)."""
    raw = json.dumps({"rag_questions": [{"question": "q?"}]})
    first = RagDataset.model_validate_json(raw)
    second = RagDataset.model_validate_json(raw)
    assert (first.rag_questions[0].question_id
            != second.rag_questions[0].question_id)


# ---------------------------------------------------------- dataset


def test_load_dataset_types_and_errors(tmp_path: Path) -> None:
    """Answered items keep their sources, bad files give ValueError."""
    good = tmp_path / "answered.json"
    good.write_text(json.dumps({"rag_questions": [{
        "question_id": "q1", "question": "q?", "answer": "a",
        "sources": [{"file_path": "a.md", "first_character_index": 0,
                     "last_character_index": 5}]}]}))
    item = load_dataset(str(good)).rag_questions[0]
    assert isinstance(item, AnsweredQuestion)
    assert item.sources[0].file_path == "a.md"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json")
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"a": 1}))
    for path in (tmp_path / "missing.json", bad_json, wrong):
        with pytest.raises(ValueError):
            load_dataset(str(path))


# ------------------------------------------------------- end to end


def test_end_to_end(tmp_path: Path) -> None:
    """Index a tiny corpus, search a dataset, evaluate the output."""
    raw = tmp_path / "raw"
    (raw / "docs").mkdir(parents=True)
    lora = "# LoRA adapters\n\nUse lora adapters to serve many models.\n"
    (raw / "docs" / "lora.md").write_text(lora, encoding="utf-8")
    (raw / "docs" / "cache.md").write_text(
        "# Prefix caching\n\nCache reuses prefix blocks.\n")
    (raw / "tool.py").write_text("def serve():\n    return 1\n")
    proc = tmp_path / "proc"
    assert run_index(str(raw), str(proc), 500) >= 3
    top = make_retriever(str(proc)).search("lora adapters", 3)
    assert top[0].file_path.endswith("lora.md")
    truth = src((raw / "docs" / "lora.md").as_posix(), 0, len(lora))
    dataset = tmp_path / "dataset.json"
    dataset.write_text(make_dataset([("q1", [truth])])
                       .model_dump_json())
    questions = json.loads(dataset.read_text())
    questions["rag_questions"][0]["question"] = "lora adapters"
    dataset.write_text(json.dumps(questions))
    out = run_search_dataset(str(dataset), 3, str(tmp_path / "out"),
                             str(proc))
    assert out == tmp_path / "out" / "dataset.json"
    results = StudentSearchResults.model_validate_json(out.read_text())
    assert results.k == 3
    scores = evaluate_results(results, load_dataset(str(dataset)), [1, 3])
    assert scores == {1: 1.0, 3: 1.0}
