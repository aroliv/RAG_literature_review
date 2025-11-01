# app.py — RAG para Revisão Narrativa (Streamlit + FAISS + Sentence-Transformers + OpenAI/Gemini)
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

# ====== Gemini (fallback opcional) ======
USE_GEMINI = False
try:
    import google.generativeai as genai  # pip install google-generativeai>=0.7.2
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False

# ====== Exportação Word (opcional) ======
try:
    from docx import Document  # pip install python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

DEFAULT_EMB = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# =========================
# Estruturas e utilitários
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
    """Lê PDF e retorna lista de textos por página e o título (nome do arquivo)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=path_or_bytes, filetype="pdf")
        title = "uploaded.pdf"
    else:
        doc = fitz.open(path_or_bytes)
        title = os.path.basename(path_or_bytes)
    pages = [p.get_text("text") for p in doc]
    return pages, title

def build_citation(ch: Chunk) -> str:
    """Gera marcador [arquivo:pag] usado no contexto; no texto, o modelo cita (arquivo:pag)."""
    base = os.path.splitext(os.path.basename(ch.title))[0]
    return f"[{base}:{ch.page_start+1}]"

def files_used_from_chunks(chunks: List[Chunk]) -> List[str]:
    """Retorna os nomes-base (sem extensão) dos arquivos usados."""
    names = {os.path.splitext(os.path.basename(ch.title))[0] for ch in chunks}
    return sorted(names)


# =========================
# Chunking e índice vetorial
# =========================

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


# =========================
# Prompt — revisão narrativa
# =========================

def make_prompt(theme: str, selected: List[Chunk]) -> str:
    """
    Prompt para revisão narrativa coesa, citando somente os PDFs enviados.
    Citações no corpo: (arquivo:pag). Sem autores/APA. Prosa contínua (sem bullets).
    """
    context_blocks = [f"{build_citation(ch)}\n{ch.text}" for ch in selected]
    context = "\n\n".join(context_blocks)

    return (
        "Você é um pesquisador sênior escrevendo uma **revisão de literatura narrativa**.\n"
        "Responda **somente** com base nos trechos fornecidos; não use fontes externas.\n"
        "Escreva em **prosa contínua**, acadêmica e coesa, **sem listas ou bullets**.\n"
        "Use **conectivos e transições** para encadear ideias entre parágrafos (por exemplo: "
        "'em síntese', 'por outro lado', 'em consonância com', 'à luz desses achados').\n"
        "As citações devem aparecer **no corpo do texto** no formato **(arquivo:pag)** — ex.: (Smith2020:12). "
        "Não cite autores/anos; **apenas** o nome do arquivo e a página.\n\n"
        f"TEMA: {theme}\n\n"
        "=== TRECHOS DISPONÍVEIS (com [arquivo:pag]) ===\n"
        f"{context}\n\n"
        "=== PRODUZIR ===\n"
        "Um texto em português com 700–1100 palavras, em parágrafos fluentes (sem tópicos), que:\n"
        "• introduz o tema e a relevância;\n"
        "• encadeia argumentos, comparando convergências e divergências entre os trechos;\n"
        "• aponta limitações e oportunidades futuras;\n"
        "• emprega citações (arquivo:pag) no corpo quando apropriado.\n"
    )


# =========================
# Chamadas LLM
# =========================

def call_openai(prompt: str, model_name: str = "gpt-4o-mini") -> str:
    """Gera conteúdo via OpenAI; defina OPENAI_MODEL='gpt-5' se tiver acesso."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Você é um assistente de revisão acadêmica e deve se ater estritamente ao RAG fornecido."},
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
    if requested in available: return requested
    prefixed = f"models/{requested}"
    if prefixed in available: return prefixed
    if requested.startswith("models/") and requested[7:] in available: return requested[7:]
    for cand in ["gemini-1.5-flash", "models/gemini-1.5-flash", "gemini-1.5-pro", "models/gemini-1.5-pro"]:
        if cand in available: return cand
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
        st.error(f"Erro ao chamar o Gemini: {e}")
        return "Falha ao gerar revisão com Gemini."


# =========================
# Fallback sem LLM
# =========================

def extractive_review(theme: str, selected: List[Chunk]) -> str:
    """Fallback simples: concatena trechos com marcação (arquivo:pag)."""
    lines = [f"Revisão (extrativa) — {theme}"]
    for ch in selected:
        lines.append(f"{ch.text.strip()} ({os.path.splitext(os.path.basename(ch.title))[0]}:{ch.page_start+1})")
    return "\n\n".join(lines)


# =========================
# Exportação
# =========================

def to_markdown(theme: str, body: str, used_files: List[str]) -> str:
    used = "\n".join(f"- {u}" for u in used_files)
    return (
        f"# Revisão de Literatura — {theme}\n\n"
        + body
        + "\n\n## Arquivos utilizados\n"
        + (used if used else "*Nenhum*")
    )

def to_docx(theme: str, body_text: str, used_files: List[str]) -> bytes:
    doc = Document()
    doc.add_heading(f"Revisão de Literatura — {theme}", level=0)
    for para in body_text.split("\n\n"):
        p = para.strip()
        if not p:
            doc.add_paragraph("")
        else:
            doc.add_paragraph(p)
    doc.add_heading("Arquivos utilizados", level=1)
    for f in used_files:
        doc.add_paragraph(f"- {f}")
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


# =========================
# UI — Streamlit
# =========================

st.set_page_config(page_title="RAG Review — Revisão narrativa", layout="wide")
st.title("RAG para Revisão Narrativa — múltiplos PDFs ➜ texto coeso com (arquivo:pag)")

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
    provider = st.selectbox(
        "Provedor",
        ["auto (OpenAI→Gemini→Extrativo)", "OpenAI", "Gemini", "Somente extrativo"],
        index=0
    )

    st.divider()
    st.subheader("OpenAI")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))  # ou "gpt-5" se tiver acesso
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

# Ingestão a partir dos uploads
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
    st.success(f"Indexados {len(st.session_state.chunks)} trechos de {len(uploaded)} PDFs.")

# Execução
if run_btn:
    if not theme.strip():
        st.warning("Informe um tema/pergunta de pesquisa.")
    elif st.session_state.vector is None:
        st.warning("Envie os PDFs antes.")
    else:
        with st.spinner("Buscando e sintetizando..."):
            hits = st.session_state.vector.search(theme, k=int(topk))
            selected = diversify_by_doc(hits, per_doc=int(per_doc), max_total=int(max_total))

            # Auditoria opcional dos trechos
            with st.expander("Trechos selecionados (auditoria)"):
                for ch in selected:
                    st.markdown(f"**{build_citation(ch)} — {ch.title}**")
                    st.write(ch.text)
                    st.divider()

            prompt = make_prompt(theme, selected)

            # Provedor
            body_text = ""
            choice = provider.lower()
            use_openai = (("openai" in choice and USE_OPENAI and os.getenv("OPENAI_API_KEY"))
                          or (choice.startswith("auto") and USE_OPENAI and os.getenv("OPENAI_API_KEY")))
            use_gemini = (("gemini" in choice and USE_GEMINI and gemini_key)
                          or (choice.startswith("auto") and (not use_openai) and USE_GEMINI and gemini_key))

            if use_openai:
                model = os.getenv("OPENAI_MODEL", openai_model or "gpt-4o-mini")
                body_text = call_openai(prompt, model_name=model)
            elif use_gemini:
                body_text = call_gemini(prompt, model_name=gemini_model)
            else:
                body_text = extractive_review(theme, selected)

            used_files = files_used_from_chunks(selected)

            st.markdown("### Revisão gerada")
            st.write(body_text)

            st.markdown("### Arquivos utilizados")
            for f in used_files:
                st.write(f"- {f}")

            # Exportação
            st.markdown("---")
            st.subheader("Exportar")
            md = to_markdown(theme, body_text, used_files)
            st.download_button(
                "Baixar Markdown (.md)",
                data=md.encode("utf-8"),
                file_name="revisao.md",
                mime="text/markdown"
            )
            if HAS_DOCX:
                docx_bytes = to_docx(theme, body_text, used_files)
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
**Observações**
- O texto é narrativo e coeso; as citações aparecem no corpo como (arquivo:pag).
- A seção final “Arquivos utilizados” serve para você montar as referências manualmente.
- Se os PDFs forem escaneados sem texto, rode OCR (ex.: Tesseract/ocrmypdf) antes.
"""
)



