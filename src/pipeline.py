from .corpus_reader import build_chunks
from .tokenizer import build_index
from .data_loader import load_all, save_all
from .retriever import Retriever
from .models import RagDataset, MinimalSearchResults, StudentSearchResults
from pathlib import Path
from .bm25_experiment import BM25Experiment

import tqdm

def run_index(raw_dir: str, folder: str, size: int) -> int:
    chunks = build_chunks(raw_dir, size)
    index = build_index(chunks)
    save_all(chunks, index, folder, size)
    return len(chunks)

def make_retrieve(folder: str) -> Retriever:
    chunks, index, _ = load_all(folder)
    return Retriever(index, chunks)

def load_dataset(path: str) -> RagDataset:
    try:
        text = Path(path).read_text(encoding="utf-8")
        return RagDataset.model_validate_json(text)
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read dataset: {e}")

def run_search_dataset(dataset_path: str, k: int, save_directory: str, processed_dir: str) -> Path:
    dataset = load_dataset(dataset_path)
    retriever = make_retrieve(processed_dir)
    results = [
        MinimalSearchResults(
            question_id=data.question_id,
            question=data.question,
            retrieved_sources=retriever.search(data.question, k)
        )
        for data in tqdm.tqdm(dataset.rag_questions, desc="Searching")
    ]
    output = StudentSearchResults(search_results=results, k=k)
    target = Path(save_directory) / "student_search_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    
    return target



def run_search_dataset_bm25(
    dataset_path: str,
    k: int,
    save_directory: str,
    processed_dir: str,
) -> Path:
    dataset = load_dataset(dataset_path)
    chunks, _, _ = load_all(processed_dir)

    retriever = BM25Experiment(chunks)

    results = [
        MinimalSearchResults(
            question_id=data.question_id,
            question=data.question,
            retrieved_sources=retriever.search(data.question, k),
        )
        for data in tqdm.tqdm(dataset.rag_questions, desc="BM25 search")
    ]

    output = StudentSearchResults(search_results=results, k=k)

    target = Path(save_directory) / "bm25_student_search_results.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        output.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return target
