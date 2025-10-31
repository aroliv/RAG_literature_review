# app.py — Streamlit RAG para revisão bibliográfica a partir de múltiplos PDFs
# Executar: streamlit run app.py

from __future__ import annotations
import os
import io
import re
import time
import base64
from dataclasses import dataclass
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz

# === (opcional) Gemini ===
USE_GEMINI = False
try:
    import google.generativeai as genai
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False

# === (opcional) exportar DOCX ===
try:
    from docx import Document
    from docx.shared import Pt
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

# ----------------------------
# Utilitários
# ----------------------------

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    title: str


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def read_pdf(path_or_bytes: bytes | str) -> Tuple[List[str], str]:
    """Lê PDF, retorna lista de textos por página e título (filename)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
        title = "uploaded.pdf"
    else:
        doc = fitz.open(path_or_bytes)
        title = os.path.basename(path_or_bytes)
    pages = []
    for p in doc:
        pages.append(p.get_text("text"))
    return pages, title


def chunk_pages(pages: List[str], title: str, doc_id: str, chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    """Cria chunks deslizantes sobre o texto concatenado, preservando mapeamento aproximado de páginas."""
    joined = "\n".join(pages)
    tokens = joined.split()

    # mapa de token_index -> page
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

    chunks: List[Chunk] = []
    start = 0
    cid = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        text = clean_text(" ".join(tokens[start:end]))
        p_start = token_index_to_page(start)
        p_end = token_index_to_page(end)
        chunks.append(Chunk(doc_id=doc_id, chunk_id=f"{doc_id}-{cid}", text=text, page_start=p_start, page_end=p_end, title=title))
        cid += 1
        if end == len(tokens):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


class VectorIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.emb_model = SentenceTransformer(model_name)
        self.index: faiss.Index = None
        self.vectors: np.ndarray | None = None
        self.meta: List[Chunk] = []

    def add(self, chunks: List[Chunk]):
        embs = self.emb_model.encode([c.text for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.vectors = embs
        else:
            self.vectors = np.vstack([self.vectors, embs])
        self.meta.extend(chunks)
        self.index.add(embs)

    def search(self, query: str, k: int = 20) -> List[Tuple[Chunk, float]]:
        q = self.emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.meta[idx], float(score)))
        return results


def diversify_by_doc(hits: List[Tuple[Chunk, float]], per_doc: int = 3, max_total: int = 12) -> List[Chunk]:
    bydoc: Dict[str, List[Tuple[Chunk, float]]] = {}
    for ch, sc in hits:
        bydoc.setdefault(ch.doc_id, []).append((ch, sc))
    for d in bydoc:
        bydoc[d].sort(key=lambda x: x[1], reverse=True)
    # round-robin
    out: List[Chunk] = []
    round_i = 0
    while len(out) < max_total:
        added = 0
        for d in sorted(bydoc.keys()):
            if round_i < len(bydoc[d]) and round_i < per_doc:
                out.append(bydoc[d][round_i][0])
                added += 1
                if len(out) >= max_total:
                    break
        if added == 0:
            break
        round_i += 1
    return out


def build_citation(ch: Chunk) -> str:
    base = os.path.splitext(os.path.basename(ch.title))[0]
    return f"[{base}:{ch.page_start+1}]"


def make_prompt(theme: str, selected: List[Chunk]) -> str:
    context_blocks = []
    for ch in selected:
        context_blocks.append(f"{build_citation(ch)}\n{ch.text}")
    context = "\n\n".join(context_blocks)
    return (
        "Você é um assistente de revisão sistemática em marketing/gestão/consumo.\n"
        "Responda APENAS com base nos trechos fornecidos; se faltar evidência, diga 'não há suporte nos trechos'.\n"
        "Use seções com títulos claros e bullets quando útil.\n"
        "Inclua citações no formato [doc:pag].\n\n"
        f"TEMA / PERGUNTA DE PESQUISA: {theme}\n\n"
        "TRECHOS (com citações):\n"
        f"{context}\n\n"
        "GERAR (em português):\n"
        "# 1. Contexto e definições\n"
        "# 2. Métodos predominantes (amostras, delineamentos)\n"
        "# 3. Achados-chave (com bullets e citações)\n"
        "# 4. Lacunas e controvérsias\n"
        "# 5. Agenda de pesquisa (3–7 proposições testáveis)\n"
        "# 6. Limitações do corpus analisado (baseado nos trechos)\n"
    )


def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    resp = genai.GenerativeModel(model_name).generate_content(prompt)
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # fallback
    return "Não foi possível gerar texto com o Gemini nesta tentativa."


def extractive_review(theme: str, selected: List[Chunk]) -> str:
    lines = [f"## Revisão (extrativa) — {theme}"]
    for ch in selected:
        lines.append(f"- {ch.text.strip()} {build_citation(ch)}")
    return "\n".join(lines)


def to_markdown(theme: str, body: str, mapping: Dict[str, str]) -> str:
    refs = [f"- [{k}] {v}" for k, v in sorted(mapping.items())]
    return (
        f"# Revisão de Literatura — {theme}\n\n" + body + "\n\n" + "## Referências (arquivo:base)\n" + "\n".join(refs)
    )


def to_docx(theme: str, body_md: str) -> bytes:
    # Conversão simples: escrever como texto corrido (sem parse de Markdown sofisticado).
    doc = Document()
    doc.add_heading(f"Revisão de Literatura — {theme}", level=0)
    for line in body_md.splitlines():
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("- "):
            p = doc.add_paragraph(line[2:].strip(), style=None)
        else:
            if line.strip() == "":
                doc.add_paragraph("")
            else:
                doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="RAG Review – PDFs ➜ Revisão", layout="wide")
st.title("RAG para Revisão Bibliográfica — múltiplos PDFs ➜ Revisão estruturada")

with st.sidebar:
    st.header("Parâmetros")
    chunk_size = st.number_input("Tamanho do chunk (tokens ~ palavras)", 400, 3000, 1200, step=100)
    overlap = st.number_input("Sobreposição", 0, 800, 200, step=50)
    topk = st.slider("Top‑K inicial (busca)", 5, 100, 30, step=5)
    per_doc = st.slider("Diversidade por documento (máx. chunks)", 1, 10, 3, step=1)
    max_total = st.slider("Máximo total de chunks no prompt", 5, 30, 12, step=1)

    st.subheader("Gemini")
    gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    gemini_model = st.selectbox("Modelo", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    if gemini_key and USE_GEMINI:
        genai.configure(api_key=gemini_key)

uploaded = st.file_uploader("Envie múltiplos PDFs", type=["pdf"], accept_multiple_files=True)

st.markdown("---")
col1, col2 = st.columns([2,1])
with col1:
    theme = st.text_area("Tema / pergunta de pesquisa", placeholder="ex.: Impacto de IA generativa na percepção de autenticidade de marcas")
with col2:
    run_btn = st.button("Gerar revisão", type="primary")

# Estado
if "vector" not in st.session_state:
    st.session_state.vector = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "chunks" not in st.session_state:
    st.session_state.chunks: List[Chunk] = []

# Ingestão em memória a partir dos uploads
if uploaded and st.session_state.vector is None:
    with st.spinner("Processando PDFs e criando índice..."):
        vec = VectorIndex()
        mapping: Dict[str, str] = {}
        all_chunks: List[Chunk] = []
        for i, uf in enumerate(uploaded):
            raw = uf.read()
            pages, title = read_pdf(raw)
            doc_id = f"doc{i+1}"
            mapping[os.path.splitext(title)[0]] = title
            cs = chunk_pages(pages, title=title, doc_id=doc_id, chunk_size=int(chunk_size), overlap=int(overlap))
            all_chunks.extend(cs)
        vec.add(all_chunks)
        st.session_state.vector = vec
        st.session_state.mapping = mapping
        st.session_state.chunks = all_chunks
    st.success(f"Indexados {len(st.session_state.chunks)} chunks de {len(uploaded)} PDFs.")

# Execução da revisão
if run_btn:
    if not theme.strip():
        st.warning("Informe um tema/pergunta de pesquisa.")
    elif st.session_state.vector is None:
        st.warning("Envie os PDFs antes.")
    else:
        with st.spinner("Buscando e sintetizando..."):
            hits = st.session_state.vector.search(theme, k=int(topk))
            selected = diversify_by_doc(hits, per_doc=int(per_doc), max_total=int(max_total))

            # Prévia dos trechos usados
            with st.expander("Trechos selecionados (para auditoria)"):
                for ch in selected:
                    st.markdown(f"**{build_citation(ch)} — {ch.title}**")
                    st.write(ch.text)
                    st.divider()

            body_text = ""
            if gemini_key and USE_GEMINI:
                prompt = make_prompt(theme, selected)
                body_text = call_gemini(prompt, model_name=gemini_model)
            else:
                body_text = extractive_review(theme, selected)

            md = to_markdown(theme, body_text, st.session_state.mapping)
            st.markdown("### Revisão gerada")
            st.write(body_text)

            # Downloads
            st.markdown("---")
            st.subheader("Exportar")
            st.download_button("Baixar Markdown (.md)", data=md.encode("utf-8"), file_name="revisao.md", mime="text/markdown")
            if HAS_DOCX:
                docx_bytes = to_docx(theme, body_text)
                st.download_button("Baixar Word (.docx)", data=docx_bytes, file_name="revisao.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            else:
                st.caption("python-docx não disponível — instale para exportar .docx")

st.markdown(
    """
---
**Dicas**
- Se o PDF for escaneado sem texto, faça OCR antes (ex.: Tesseract).
- Ajuste *Top‑K* e *Diversidade por documento* para controlar foco e abrangência.
- Para maior qualidade, troque o embedding para `all-MiniLM-L12-v2` (pode ficar mais pesado). 
- O app mantém tudo em memória; se quiser persistir um índice, me peça que eu adapto para salvar/recuperar FAISS em disco.
"""
)
