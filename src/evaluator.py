from .models import MinimalSource, RagDataset, StudentSearchResults, AnsweredQuestion

THRESHOLD = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Overlap of two character ranges, as intersection over union."""
    inter = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if inter <= 0:
        return 0.0
    width_a = a.last_character_index - a.first_character_index
    width_b = b.last_character_index - b.first_character_index
    return inter / (width_a + width_b - inter)


def is_found(truth: MinimalSource, retrieved: list[MinimalSource]) -> bool:
    """Tell whether one of ``retrieved`` matches the correct source."""
    return any(
        item.file_path == truth.file_path and iou(item, truth) >= THRESHOLD
        for item in retrieved
    )


def question_recall(
    truth: list[MinimalSource], retrieved: list[MinimalSource], k: int
) -> float:
    """Share of the correct sources found in the top k results."""
    top = retrieved[:k]
    return sum(is_found(source, top) for source in truth) / len(truth)


def evaluate_results(
    results: StudentSearchResults, dataset: RagDataset, ks: list[int]
) -> dict[int, float]:
    """Average recall@k over every question that has correct sources."""
    truth = {
        q.question_id: q.sources
        for q in dataset.rag_questions
        if isinstance(q, AnsweredQuestion) and q.sources
    }
    found = {r.question_id: r.retrieved_sources for r in results.search_results}
    totals = {k: 0.0 for k in ks}
    for question_id, sources in truth.items():
        for k in ks:
            totals[k] += question_recall(sources, found.get(question_id, []), k)
    count = len(truth)
    return {k: totals[k] / count if count else 0.0 for k in ks}