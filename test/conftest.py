"""Shared fixtures."""
from typing import Callable

import pytest

from src.tokenizer import build_index
from src.models import Chunk
from src.retriever import Retriever

MakeChunks = Callable[[list[str]], list[Chunk]]

FOUR_TEXTS = [
    "load model config",
    "server server model config",
    "start model server",
    "load weights from disk",
]


@pytest.fixture
def make_chunks() -> MakeChunks:
    """Return a factory building chunks f0.md, f1.md... from texts."""

    def build(texts: list[str]) -> list[Chunk]:
        return [
            Chunk(
                file_path=f"f{i}.md",
                first_character_index=0,
                last_character_index=max(len(text), 1),
                text=text,
            )
            for i, text in enumerate(texts)
        ]

    return build


@pytest.fixture
def four_chunks(make_chunks: MakeChunks) -> list[Chunk]:
    """The four-chunk example used in the BM25 walkthrough."""
    return make_chunks(FOUR_TEXTS)


@pytest.fixture
def retriever(four_chunks: list[Chunk]) -> Retriever:
    """A retriever over the four-chunk example."""
    return Retriever(build_index(four_chunks), four_chunks)
