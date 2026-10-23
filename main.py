from src.corpus_reader import build_chunks
from src.tokenizer import tokenize, build_index
from src.data_loader import save_all, load_all

chunks = build_chunks("data/raw", 2000)
index = build_index(chunks)

save_all(chunks, index, "data/processed", 2000)
l_chunks, l_index, meta = load_all("data/processed")

assert l_chunks == chunks
assert l_index == index


