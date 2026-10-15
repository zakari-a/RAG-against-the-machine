from src.corpus_read.corpus_reader import build_chunks
from src.tokenizer import tokenize, build_index

chunks = build_chunks("data/raw", 500)
index = build_index(chunks[:3])

print(index.postings)


