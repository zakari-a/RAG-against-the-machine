from pathlib import Path
from tqdm import tqdm
from src.models import Chunk
from src.md_chunker import chunk_markdown, chunk_python, chunk_text


def list_files(root: str, suffixs: list[str]) -> list[Path]:
    base = Path(root)
    if not base.is_dir():
        raise FileNotFoundError(f"corpus folder not found {root}")

    paths = []
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixs:
            paths.append(path)
    return sorted(paths)

def read_files(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", newline="") as file:
            return file.read()
    except (UnicodeDecodeError, OSError):
        return None

def build_chunks(root: str, size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    skipped = 0
    for path in tqdm(list_files(root, [".py", ".md", ".txt"])):
        text = read_files(path)
        if text is None:
            skipped += 1
            continue
        if path.suffix == ".md":
            chunks.extend(chunk_markdown(path.as_posix(), text, size))
        elif path.suffix == ".py":
            chunks.extend(chunk_python(path.as_posix(), text, size))
        elif path.suffix == ".txt":
            chunks.extend(chunk_text(path.as_posix(), text, size))

    print(f"{len(chunks)} chunks, {skipped} files skipped")
    return chunks
