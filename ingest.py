# ingest.py - cria indice FAISS em data/index a partir de data/papers
from __future__ import annotations
import os, re, pickle
from typing import List, Tuple
import fitz
from sentence_transformers import SentenceTransformer
import faiss
DATA_DIR = "data/papers"
INDEX_DIR = "data/index"
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
def clean_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s).strip()
def read_pdf(path: str) -> Tuple[list[str], str]:
    doc = fitz.open(path)
    pages = [p.get_text('text') for p in doc]
    title = os.path.basename(path)
    return pages, title
def chunk_pages(pages: list[str], title: str, chunk_size: int, overlap: int) -> list[dict]:
    joined = '\n'.join(pages)
    tokens = joined.split()
    page_offsets = []
    tok_count = 0
    for i, page in enumerate(pages):
        n = len(page.split())
        page_offsets.append((tok_count, i))
        tok_count += n
    def token_index_to_page(idx: int) -> int:
        best = 0
        for (start_tok, page_idx) in page_offsets:
            if idx >= start_tok:
                best = page_idx
            else:
                break
        return best
    chunks = []
    start = 0
    cid = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        text = clean_text(' '.join(tokens[start:end]))
        p_start = token_index_to_page(start)
        p_end = token_index_to_page(end)
        chunks.append({'text': text, 'page_start': p_start, 'page_end': p_end, 'title': title, 'chunk_id': cid})
        cid += 1
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks
def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    model = SentenceTransformer(EMB_MODEL)
    all_chunks = []
    titles = []
    for fname in os.listdir(DATA_DIR):
        if not fname.lower().endswith('.pdf'):
            continue
        path = os.path.join(DATA_DIR, fname)
        pages, title = read_pdf(path)
        chs = chunk_pages(pages, title, CHUNK_SIZE, CHUNK_OVERLAP)
        for c in chs:
            c['doc'] = fname
        titles.append(title)
        all_chunks.extend(chs)
    if not all_chunks:
        print('Nenhum PDF encontrado em data/papers/')
        return
    texts = [c['text'] for c in all_chunks]
    embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, os.path.join(INDEX_DIR, 'index.faiss'))
    with open(os.path.join(INDEX_DIR, 'meta.pkl'), 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f'Indexados {len(all_chunks)} chunks de {len(titles)} PDFs. Saida em data/index/')
if __name__ == '__main__':
    main()