"""Compare search results at k=5 (throwaway diagnostic, not submitted).

Usage:
    python compare.py RESULTS_A RESULTS_B ANSWERED_DATASET
    python compare.py RESULTS_A ANSWERED_DATASET

With two runs it prints the sources found by one run and not the other.
With one run it prints the sources that run missed.
"""
import sys
from pathlib import Path

from src.evaluator import is_found
from src.models import (AnsweredQuestion, MinimalSource, RagDataset,
                        StudentSearchResults)
from src.pipeline import load_dataset

Key = tuple[str, int]
Info = dict[Key, tuple[str, MinimalSource]]


def load_results(path: str) -> StudentSearchResults:
    """Read a search results file, or raise ValueError."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        return StudentSearchResults.model_validate_json(text)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read results: {path}") from error


def source_info(dataset: RagDataset) -> Info:
    """Map (question id, source number) to its question and source."""
    info: Info = {}
    for q in dataset.rag_questions:
        if isinstance(q, AnsweredQuestion):
            for i, truth in enumerate(q.sources):
                info[(q.question_id, i)] = (q.question, truth)
    return info


def found_at_5(results: StudentSearchResults, info: Info) -> dict[Key, bool]:
    """Tell, for every correct source, whether the top 5 found it."""
    got = {r.question_id: r.retrieved_sources[:5]
           for r in results.search_results}
    return {
        key: is_found(truth, got.get(key[0], []))
        for key, (_, truth) in info.items()
    }


def per_extension(flags: dict[Key, bool], info: Info) -> dict[str, str]:
    """Count found/total sources for each file extension."""
    totals: dict[str, list[int]] = {}
    for key, found in flags.items():
        ext = Path(info[key][1].file_path).suffix or "(none)"
        pair = totals.setdefault(ext, [0, 0])
        pair[0] += int(found)
        pair[1] += 1
    return {ext: f"{p[0]}/{p[1]}" for ext, p in sorted(totals.items())}


def describe(key: Key, info: Info) -> str:
    """Two readable lines about one correct source."""
    question, truth = info[key]
    span = f"[{truth.first_character_index}:{truth.last_character_index}]"
    return f"{truth.file_path} {span}\n    {question[:100]}"


def report(name: str, flags: dict[Key, bool], info: Info) -> None:
    """Print how many sources a run found, and by extension."""
    found = sum(flags.values())
    print(f"{name}: {found}/{len(flags)} sources found in the top 5 "
          f"({found / len(flags):.1%})")
    print(f"  by extension: {per_extension(flags, info)}")


def show_flipped(a: dict[Key, bool], b: dict[Key, bool], info: Info) -> None:
    """Print the sources on which the two runs disagree."""
    flips = [key for key in a if a[key] != b[key]]
    print(f"\n{len(flips)} source(s) differ between A and B")
    for key in flips:
        print(f"  A={a[key]} B={b[key]} {describe(key, info)}")


def show_missed(flags: dict[Key, bool], info: Info, limit: int = 15) -> None:
    """Print the sources a single run did not find."""
    missed = [key for key, found in flags.items() if not found]
    print(f"\n{len(missed)} source(s) not found; first {limit}:")
    for key in missed[:limit]:
        print(f"  {describe(key, info)}")


def main(argv: list[str]) -> int:
    """Run the comparison; return a process exit code."""
    if len(argv) not in (3, 4):
        print("usage: compare.py RESULTS_A [RESULTS_B] ANSWERED_DATASET")
        return 1
    paths = argv[1:-1]
    try:
        info = source_info(load_dataset(argv[-1]))
        runs = [load_results(path) for path in paths]
    except ValueError as error:
        print(f"error: {error}")
        return 1
    if not info:
        print("error: no correct sources found; pass the AnsweredQuestions "
              "dataset, not the unanswered one")
        return 1
    for path, run in zip(paths, runs):
        if run.k < 5:
            print(f"warning: {path} was searched with k={run.k}, below 5")
    flags = [found_at_5(run, info) for run in runs]
    for path, flag in zip(paths, flags):
        report(path, flag, info)
    if len(flags) == 2:
        show_flipped(flags[0], flags[1], info)
    else:
        show_missed(flags[0], info)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))