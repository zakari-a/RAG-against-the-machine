import fire
from rich import print
from pathlib import Path

from .models import StudentSearchResults
from .pipeline import run_index, make_retrieve, run_search_dataset, load_dataset, run_search_dataset_bm25
from .evaluator import evaluate_results

class Cli():

    def index(self, max_chunk_size: int=2000,
              processed_dir: str="data/processed",
              raw_dir: str="data/raw") -> None:
        if isinstance(max_chunk_size, bool) or not \
            isinstance(max_chunk_size, int) or not \
                0 < max_chunk_size <= 2000:
            print(
                "max_chunk_size should be a positive int "
                "(0 < max_chunk_size <= 2000)")
            return
        try:
            total = run_index(raw_dir, processed_dir, max_chunk_size)
        except (OSError, ValueError) as e:
            print(e)
            return

        print(f"Ingestion complete! Indexed {total} chunks under {processed_dir}/")
            

    def search(self, query: str, k: int = 10, processed_dir: str = "data/processed") -> None:
        query = str(query)
        if not query.strip():
            print(f"query is empty")
            return
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            print("k should be a positive int")
            return 
        
        try:
            retriever = make_retrieve(processed_dir)
        except(OSError, ValueError, RuntimeError) as e:
            print(f"error: {e}")
            return

        for src in retriever.search(query, k):
            print(f"{src.file_path}: [{src.first_character_index}:{src.last_character_index}]")


    def search_dataset(self, dataset_path: str, k: int, save_directory: str, processed_dir: str = "data/processed") -> None:
        if not dataset_path or not save_directory:
            print("error: --dataset_path and --save_directory are required")
            return
        if isinstance(k, bool) or k <= 0:
            print("error: k should be a positive int")
            return
        try:
            target = run_search_dataset(dataset_path, k, save_directory, processed_dir)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"error: {e}")
            return
        print(f"Saved student_search_results to {target}")


    def evaluate(
        self, student_search_results_path: str = "", dataset_path: str = ""
    ) -> None:
        """Print recall@k of a results file against a ground truth."""
        if not student_search_results_path or not dataset_path:
            print("error: both paths are required")
            return
        try:
            text = Path(student_search_results_path).read_text(encoding="utf-8")
            results = StudentSearchResults.model_validate_json(text)
            dataset = load_dataset(dataset_path)
        except (OSError, ValueError) as error:
            print(f"error: {error}")
            return
        ks = [k for k in (1, 3, 5, 10) if k <= results.k]
        for k, value in evaluate_results(results, dataset, ks).items():
            print(f"Recall@{k}: {value:.3f} ({value:.1%})")



if __name__=="__main__":
    fire.Fire(Cli)