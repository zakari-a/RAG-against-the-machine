from .tokenizer import InvertedIndex, tokenize
from .models import Chunk, MinimalSource
import math
import heapq

class Retriever():
    def __init__(self, index: InvertedIndex, chunks: list[Chunk],
                k1: float = 1.5, b: float = 0.75) -> None:
        self.index = index
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.n = len(chunks)
        total = sum(index.lenghts)
        self.avg_lenght = total / self.n if self.n else 0.0
        
    def idf(self, term: str) -> float:
        postings = self.index.postings.get(term)
        if not postings:
            return 0.0
        df = len(postings)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def term_score(self, idf: float, count: int, chunk_id: int) -> float:
        lenght = self.index.lenghts[chunk_id]
        norm = self.k1 * (1 - self.b + self.b * lenght / self.avg_lenght)
        return idf * count * (self.k1 + 1) / (count + norm)

    def score(self, query: str):
        scores: dict[int, float] = {}
        for term in sorted(set(tokenize(query))):
            postings = self.index.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)

            for chunk_id, count in postings:
                score = self.term_score(idf, count, chunk_id)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score
        return scores

    def search(self, query: str, k: int) -> list[MinimalSource]:
        if k <= 0:
            return []
        scores = self.score(query)
        best = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [self.chunks[chunk_id].to_source() for chunk_id, _ in best]

