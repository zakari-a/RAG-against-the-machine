from pydantic import BaseModel
from .md_chunker import chunk_markdown, chunk_python, chunk_text
from .models import Chunk
from .tokenizer import InvertedIndex

from pathlib import Path
import json

CHUNKERS = {
    ".py": chunk_python,
    ".md": chunk_markdown,
    ".txt": chunk_text
}

class Meta(BaseModel):
    max_chunk_size: int
    num_chunks: int
    exstensions: list[str]


def save_all(chunks: list[Chunk], index: IndentationError, folder: str, size: int) -> None:
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump() for chunk in chunks]
    (root / "chunks.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (root / "index.json").write_text(index.model_dump_json(), encoding="utf-8")
    meta = Meta(
        max_chunk_size=size,
        num_chunks = len(chunks),
        exstensions=sorted(CHUNKERS)
    )
    (root / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")


def load_all(folder: str)-> tuple[list[Chunk], InvertedIndex, Meta]:
    root = Path(folder)
    try:
        raw = (root / "chunks.json").read_text(encoding="utf-8")
        chunks = [Chunk.model_validate(item) for item in json.loads(raw)]
        text = (root / "index.json").read_text(encoding="utf-8")
        index = InvertedIndex.model_validate_json(text)
        meta_text = (root / "meta.json").read_text(encoding="utf-8")
        meta = Meta.model_validate_json(meta_text)
    except(OSError, ValueError):
        raise RuntimeError("index missing or corupt:  run --index first")
    return chunks , index, meta