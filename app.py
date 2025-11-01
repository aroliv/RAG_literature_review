# app.py — RAG para Revisão Bibliográfica (Streamlit + FAISS + Sentence-Transformers + OpenAI/Gemini)
# Execução: streamlit run app.py

from __future__ import annotations
import os
import io
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss

# ====== OpenAI (principal) ======
USE_OPENAI = False
try:
    from openai import OpenAI  # pip install openai>=1.40.0
    USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# ====== Gemini (opcional / fallback) ======
USE_GEMINI = False
try:
    import google.generativeai as genai  # pip install google-generativeai>=0.7.2
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False

# ====== Exportação para Word (opcional) ======
try:
    from docx import Document  # pip install python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

DEFAULT_EMB = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# =========================
# Utilitários de dados
# =========================

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
    """Lê PDF e retorna lista de textos por página e um 'título' (nome do arquivo)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
        title = "uploaded.pdf"
    else:
        doc = fitz.open(path_or_bytes)
        title = os.path.basename(path_or_bytes)
    pages = [p.get_text("text") for p in doc]
    return pages, title

def chunk_pages(
    pages: List[str],
    title: str,
    doc_id: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> List[Chunk]:
    """Cria chunks deslizantes preservando mapeamento aproximado para páginas."""
    joined = "\n".join(pages)
    tokens = joined.split()

    # mapa token_index -> índice de página
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
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}-{cid}",
                text=text,
                page_start=p_start,
                page_end=p_end,
                title=title,
            )
        )
        cid += 1
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks

class VectorIndex:
    def __init__(self, model_name: str = DEFAULT_EMB):
        self.model_name = model_name
        self.emb_model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.vectors = None
        self.meta: List[Chunk] = []

    def add(self, chunks: List[Chunk]):
        embs = self.emb_model.encode(
            [c.text for c in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.vectors = embs
        else:
            import numpy as np
            self.vectors = np.vstack([self.vectors, embs])
        self.meta.extend(chunks)
        self.index.add(embs)

    def search(self, query: str, k: int = 20) -> List[Tuple[Chunk, float]]:
        q = self.emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results: List[Tuple[Chunk, float]] = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.meta[idx], float(score)))
        return results

def diversify_by_doc(
    hits: List[Tuple[Chunk, float]],
    per_doc: int = 3,
    max_total: int = 12
) -> List[Chunk]:
    """Round-robin por documento: limita nº de chunks por paper e total."""
    bydoc: Dict[str, List[Tuple[Chunk, float]]] = {}
    for ch, sc in hits:
        bydoc.setdefault(ch.doc_id, []).append((ch, sc))
    for d in bydoc:
        bydoc[d].sort(key=lambda x: x[1], reverse=True)

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
    """Prompt estruturado que força citações e seções."""
    context_blocks = [f"{build_citation(ch)}\n{ch.text}" for ch in selected]
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

def extractive_review(theme: str, selected: List[Chunk]) -> str:
    """Fallback sem LLM: apenas concatena trechos + citações."""
    lines = [f"## Revisão (extrativa) — {theme}"]
    for ch in selected:
        lines.append(f"- {ch.text.strip()} {build_citation(ch)}")
    return "\n".join(lines)

def to_markdown(theme: str, body: str, mapping: Dict[str, str]) -> str:
    refs = [f"- [{k}] {v}" for k, v in sorted(mapping.items())]
    return (
        f"# Revisão de Literatura — {theme}\n\n"
        + body
        + "\n\n## Referências (arquivo:base)\n"
        + "\n".join(refs)
    )

def to_docx(theme: str, body_md: str) -> bytes:
    """Exporta conteúdo simples para DOCX (sem parse avançado de Markdown)."""
    doc = Document()
    doc.add_heading(f"Revisão de Literatura — {theme}", level=0)
    for line in body_md.splitlines():
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("- "):
            doc.add_paragraph(line[2:].strip())
        else:
            if line.strip() == "":
                doc.add_paragraph("")
            else:
                doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# =========================
# Provedores LLM
# =========================

def call_openai(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """
    Gera conteúdo via OpenAI Chat Completions (model configurable).
    Você pode definir OPENAI_MODEL="gpt-5" se sua conta tiver acesso.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Você é um assistente especializado em revisão bibliográfica acadêmica. Responda apenas com base nos trechos fornecidos (RAG)."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao chamar o modelo OpenAI: {e}")
        return "Falha ao gerar revisão com OpenAI."

def _configure_gemini(api_key: str | None) -> bool:
    if not api_key:
        return False
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Falha ao configurar Gemini: {e}")
        return False

def _list_gemini_models() -> list[str]:
    try:
        models = genai.list_models()
        names = []
        for m in models:
            if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                names.append(getattr(m, "name", None))
        return [n for n in names if n]
    except Exception:
        return []

def _resolve_model_name(requested: str) -> str:
    available = _list_gemini_models()
    if not available:
        return requested
    if requested in available:
        return requested
    prefixed = f"models/{requested}"
    if prefixed in available:
        return prefixed
    if requested.startswith("models/") and requested[7:] in available:
        return requested[7:]
    for cand in ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-pro", "models/gemini-1.5-pro"]:
        if cand in available:
            return cand
    return available[0]

def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    try:
        resolved = _resolve_model_name(model_name)
        model = genai.GenerativeModel(resolved)
        resp = model.generate_content(prompt)
        if hasattr(resp, "text") and resp.text:
            return resp.text
        return "Não foi possível gerar texto com o Gemini nesta tentativa (resposta vazia)."
    except Exception as e:
        st.error(
            "Falha ao chamar o Gemini. Verifique chave/modelo/versão. "
            f"Detalhes: {e}"
        )
        return "Erro ao gerar com Gemini."

# =========================
# UI — Streamlit
# =========================

st.set_page_config(page_title="RAG Review — PDFs ➜ Revisão", layout="wide")
st.title("RAG para Revisão Bibliográfica — múltiplos PDFs ➜ Revisão estruturada")

with st.sidebar:
    st.header("Parâmetros")
    chunk_size = st.number_input(
        "Tamanho do chunk (aprox. palavras)",
        min_value=400, max_value=3000, value=int(os.getenv("CHUNK_SIZE", "1200")), step=100
    )
    overlap = st.number_input(
        "Sobreposição", min_value=0, max_value=800, value=int(os.getenv("CHUNK_OVERLAP", "200")), step=50
    )
    topk = st.slider("Top-K inicial (busca)", 5, 100, 30, step=5)
    per_doc = st.slider("Diversidade por documento (máx. chunks)", 1, 10, 3, step=1)
    max_total = st.slider("Máximo total de chunks no prompt", 5, 30, 12, step=1)

    st.subheader("Provedor LLM")
    provider = st.selectbox("Provedor", ["auto (OpenAI→Gemini→Extrativo)", "OpenAI", "Gemini", "Somente extrativo"], index=0)

    st.caption("— O app prioriza OpenAI; você pode usar OPENAI_MODEL='gpt-5' se sua conta tiver acesso.")

    st.divider()
    st.subheader("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if openai_model:
        os.environ["OPENAI_MODEL"] = openai_model

    st.subheader("Gemini (fallback)")
    gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    gemini_model = st.selectbox("GEMINI_MODEL", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    if USE_GEMINI and gemini_key:
        _configure_gemini(gemini_key)

uploaded = st.file_uploader("Envie múltiplos PDFs", type=["pdf"], accept_multiple_files=True)

st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    theme = st.text_area(
        "Tema / pergunta de pesquisa",
        placeholder="ex.: Impacto de IA generativa na percepção de autenticidade de marcas"
    )
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
            cs = chunk_pages(
                pages, title=title, doc_id=doc_id,
                chunk_size=int(chunk_size), overlap=int(overlap)
            )
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

            with st.expander("Trechos selecionados (para auditoria)"):
                for ch in selected:
                    st.markdown(f"**{build_citation(ch)} — {ch.title}**")
                    st.write(ch.text)
                    st.divider()

            # Escolha do provedor
            body_text = ""
            choice = provider.lower()
            use_openai = (("openai" in choice and USE_OPENAI and os.getenv("OPENAI_API_KEY"))
                          or (choice.startswith("auto") and USE_OPENAI and os.getenv("OPENAI_API_KEY")))
            use_gemini = (("gemini" in choice and USE_GEMINI and gemini_key)
                          or (choice.startswith("auto") and (not use_openai) and USE_GEMINI and gemini_key))

            prompt = make_prompt(theme, selected)

            if use_openai:
                model = os.getenv("OPENAI_MODEL", openai_model or "gpt-4o-mini")
                body_text = call_openai(prompt, model_name=model)
            elif use_gemini:
                body_text = call_gemini(prompt, model_name=gemini_model)
            else:
                body_text = extractive_review(theme, selected)

            md = to_markdown(theme, body_text, st.session_state.mapping)

            st.markdown("### Revisão gerada")
            st.write(body_text)

            st.markdown("---")
            st.subheader("Exportar")
            st.download_button(
                "Baixar Markdown (.md)",
                data=md.encode("utf-8"),
                file_name="revisao.md",
                mime="text/markdown"
            )
            if HAS_DOCX:
                docx_bytes = to_docx(theme, body_text)
                st.download_button(
                    "Baixar Word (.docx)",
                    data=docx_bytes,
                    file_name="revisao.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.caption("python-docx não disponível — instale para exportar .docx")

st.markdown(
    """
---
**Dicas**
- Se o PDF for escaneado sem texto, faça OCR antes (ex.: Tesseract).
- Ajuste *Top-K* e *Diversidade por documento* para controlar foco e abrangência.
- Para maior qualidade, troque o embedding para `all-MiniLM-L12-v2` (pode ficar mais pesado).
- O app mantém tudo em memória; se quiser índice persistente, use `ingest.py`.
"""
)

