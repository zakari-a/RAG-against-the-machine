import re
import tqdm
from collections import Counter
from pydantic import BaseModel
from src.models import Chunk
import bm25s

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


class InvertedIndex(BaseModel):
    postings: dict[str, list[tuple[int, int]]]
    lenghts: list[int]

def split_identifier(word: str) -> list[str]:
    parts = []
    for peice in word.split("_"):
        parts.extend(part.lower() for part in PART_RE.findall(peice))
    return parts


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in WORD_RE.findall(text):
        parts = split_identifier(word)
        if len(parts) > 1:
            tokens.append(word.lower())
        tokens.extend(parts)
    return tokens

def build_index(chunks: list[Chunk]) -> InvertedIndex:
    postings: dict[str, list[tuple[int, int]]] = {}
    lenghts: list[int] = []
    for chunk_id, chunk in enumerate(tqdm.tqdm(chunks, desc="Tokenizing")):
        tokens = tokenize(chunk.text)
        lenghts.append(len(tokens))
        for term, count in Counter(tokens).items():
            postings.setdefault(term, []).append((chunk_id, count))
    return InvertedIndex(postings=postings, lenghts=lenghts)

