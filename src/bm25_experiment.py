
import bm25s

from bm25s.tokenization import Tokenizer

from .models import Chunk, MinimalSource
from .tokenizer import tokenize


class BM25Experiment:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

        # Reuse your existing tokenizer, including identifier splitting.
        self.tokenizer = Tokenizer(
            stemmer=None,
            stopwords=[],
            splitter=tokenize,
        )

        texts = [chunk.text for chunk in chunks]

        tokenized_chunks = self.tokenizer.tokenize(
            texts,
            return_as="tuple",
        )
        self.retriever = bm25s.BM25()
        self.retriever.index(tokenized_chunks)

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        if not self.chunks or k <= 0:
            return []

        query_tokens = self.tokenizer.tokenize(
            [query],
            update_vocab=False,
            return_as="tuple",
            show_progress=False,
        )
        results, scores = self.retriever.retrieve(
            query_tokens,
            k=min(k, len(self.chunks)),
            show_progress=False,
        )

        sources = []

        for chunk_id in results[0]:
            chunk = self.chunks[int(chunk_id)]

            sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )

        return sources
