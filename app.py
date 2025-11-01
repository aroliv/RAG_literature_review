# app.py — RAG para Revisão Narrativa (Streamlit + FAISS + Sentence-Transformers + OpenAI/Gemini)
# Execução: streamlit run app.py

from __future__ import annotations
import os
import io
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

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

# ===== Inferência de Label Autor-Ano por PDF =====

def _guess_year(pages: List[str]) -> str | None:
    # tenta ano no front-matter
    head = "\n".join(pages[:2])
    m = re.search(r"\b(20\d{2}[a-z]?|19\d{2}[a-z]?)\b", head)
    if m:
        return m.group(1)
    # tenta nas últimas páginas (rodapés / refs)
    tail = "\n".join(pages[-2:]) if len(pages) > 2 else ""
    m2 = re.search(r"\b(20\d{2}[a-z]?|19\d{2}[a-z]?)\b", tail)
    return m2.group(1) if m2 else None

def _guess_first_author_surname(pages: List[str]) -> str | None:
    head = "\n".join(pages[:2])
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    # linha com vírgulas / and / & e palavras em Title Case
    for ln in lines:
        if ("," in ln or " and " in ln or " & " in ln) and len(ln) < 200:
            parts = re.split(r",| and | & ", ln)
            for p in parts:
                toks = p.strip().split()
                if len(toks) >= 1:
                    last = toks[-1]
                    if last[:1].isupper() and last.isalpha() and len(last) >= 2:
                        return last
    # fallback: "By Nome Sobrenome"
    m = re.search(r"\bby\s+([A-Z][A-Za-z\-']+)\s+([A-Z][A-Za-z\-']+)", head, flags=re.I)
    if m:
        return m.group(2)
    return None

def build_doc_citation_key(pages: List[str], filename_stem: str) -> str:
    """Gera label estilo Huang2021; fallbacks: PrimeiraPalavraTítulo+Ano; senão filename."""
    first_author = _guess_first_author_surname(pages)
    year = _guess_year(pages)
    if first_author and year:
        return f"{first_author}{year}"
    title_line = next((ln.strip() for ln in pages[0].splitlines() if ln.strip()), "")
    first_word = title_line.split()[0] if title_line else None
    if first_word and year and first_word[:1].isalpha():
        return f"{first_word}{year}"
    return filename_stem

# ===== Citação e lista de arquivos =====

def build_citation(ch: "Chunk") -> str:
    """Usa a chave autor-ano inferida por doc_id + número de página."""
    label = st.session_state.doc_labels.get(ch.doc_id) if "doc_labels" in st.session_state else None
    if not label:
        label = os.path.splitext(os.path.basename(ch.title))[0]
    return f"[{label}:{ch.page_start+1}]"

def files_used_from_chunks(chunks: List["Chunk"]) -> List[Tuple[str, str]]:
    """Retorna pares (LabelAutorAno, NomeArquivoSemExtensão)."""
    pairs = set()
    for ch in chunks:
        label = st.session_state.doc_labels.get(ch.doc_id) if "doc_labels" in st.session_state else None
        if not label:
            label = os.path.splitext(os.path.basename(ch.title))[0]
        stem = os.path.splitext(os.path.basename(ch.title))[0]
        pairs.add((label, stem))
    return sorted(pairs)

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
            self.vectors = __import__("numpy").vstack([self.vectors, embs]) if hasattr(__import__("numpy"), "vstack") else None  # fallback
            if self.vectors is None:
                import numpy as np
                self.vectors = np.vstack([self.vectors, embs])  # noqa
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

# =========================
# Seleção focada em COBERTURA + RIQUEZA
# =========================

def select_for_coverage_and_richness(
    hits: List[Tuple[Chunk, float]],
    min_per_doc: int,
    max_per_doc: int,
    max_total: int
) -> List[Chunk]:
    """
    1) Garante cobertura: tenta pegar pelo menos 'min_per_doc' chunks de cada documento presente nos top-k.
    2) Depois preenche até 'max_total' usando os melhores restantes, sem estourar 'max_per_doc' por doc.
    Objetivo: maximizar diversidade de fontes (texto rico) sem perder qualidade global.
    """
    bydoc: Dict[str, List[Tuple[Chunk, float]]] = {}
    for ch, sc in hits:
        bydoc.setdefault(ch.doc_id, []).append((ch, sc))
    for d in bydoc:
        bydoc[d].sort(key=lambda x: x[1], reverse=True)

    out: List[Chunk] = []
    count_by_doc: Dict[str, int] = {}

    # Passo A: cobertura mínima por doc
    for doc_id, arr in bydoc.items():
        take = min(min_per_doc, len(arr))
        for i in range(take):
            if len(out) >= max_total:
                break
            out.append(arr[i][0])
            count_by_doc[doc_id] = count_by_doc.get(doc_id, 0) + 1
        if len(out) >= max_total:
            break

    if len(out) >= max_total:
        return out

    # Passo B: enriquece com melhores restantes respeitando max_per_doc
    remaining = []
    for doc_id, arr in bydoc.items():
        remaining.extend([(doc_id, ch, sc) for ch, sc in arr[min_per_doc:]])
    remaining.sort(key=lambda t: t[2], reverse=True)

    for doc_id, ch, sc in remaining:
        if len(out) >= max_total:
            break
        if count_by_doc.get(doc_id, 0) >= max_per_doc:
            continue
        out.append(ch)
        count_by_doc[doc_id] = count_by_doc.get(doc_id, 0) + 1

    return out

# =========================
# Prompt — revisão narrativa (maximiza citações)
# =========================

def make_prompt(theme: str, selected: List[Chunk], min_cites_per_paragraph: int = 2) -> str:
    """
    Prosa coesa, **sem bullets**, citando com frequência no formato (Label:pag) e variando as fontes.
    Pede citação múltipla por parágrafo quando houver suporte.
    """
    context_blocks = [f"{build_citation(ch)}\n{ch.text}" for ch in selected]
    context = "\n\n".join(context_blocks)

    return (
        "Você é um pesquisador sênior escrevendo uma revisão de literatura **narrativa**.\n"
        "Responda **somente** com base nos trechos fornecidos; não use fontes externas.\n"
        "Escreva em **prosa contínua**, acadêmica e coesa, **sem listas ou bullets**.\n"
        f"Use muitas citações internas, variando as fontes, e insira **pelo menos {min_cites_per_paragraph} citações (Label:pag)** por parágrafo, "
        "sempre que houver suporte no material. Evite depender de um único documento.\n"
        "As citações aparecem **no corpo do texto** como **(Label:pag)** — ex.: (Huang2021:12). "
        "NÃO cite autores/anos fora desse formato; NÃO invente referências.\n\n"
        f"TEMA: {theme}\n\n"
        "=== TRECHOS DISPONÍVEIS (com [Label:pag]) ===\n"
        f"{context}\n\n"
        "=== PRODUZIR ===\n"
        "Um texto em português, com 900–1300 palavras, em parágrafos fluentes (sem tópicos), que:\n"
        "• introduz o tema e a relevância;\n"
        "• encadeia argumentos com **muitas citações (Label:pag)**, comparando convergências e divergências;\n"
        "• aponta limitações e oportunidades futuras;\n"
        "• evita generalizações sem suporte; quando não houver evidência suficiente, sinalize explicitamente.\n"
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
            temperature=0.35,
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
    """Fallback simples: concatena trechos com marcação (Label:pag)."""
    lines = [f"Revisão (extrativa) — {theme}"]
    for ch in selected:
        label = st.session_state.doc_labels.get(ch.doc_id) if "doc_labels" in st.session_state else os.path.splitext(os.path.basename(ch.title))[0]
        lines.append(f"{ch.text.strip()} ({label}:{ch.page_start+1})")
    return "\n\n".join(lines)

# =========================
# Exportação
# =========================

def to_markdown(theme: str, body: str, used_files: List[Tuple[str, str]]) -> str:
    used = "\n".join(f"- {label} → {stem}" for label, stem in used_files)
    return (
        f"# Revisão de Literatura — {theme}\n\n"
        + body
        + "\n\n## Arquivos utilizados (Label → Arquivo)\n"
        + (used if used else "*Nenhum*")
    )

def to_docx(theme: str, body_text: str, used_files: List[Tuple[str, str]]) -> bytes:
    doc = Document()
    doc.add_heading(f"Revisão de Literatura — {theme}", level=0)
    for para in body_text.split("\n\n"):
        p = para.strip()
        if not p:
            doc.add_paragraph("")
        else:
            doc.add_paragraph(p)
    doc.add_heading("Arquivos utilizados (Label → Arquivo)", level=1)
    for label, stem in used_files:
        doc.add_paragraph(f"- {label} → {stem}")
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# =========================
# UI — Streamlit
# =========================

st.set_page_config(page_title="RAG Review — Revisão narrativa (rica em citações)", layout="wide")
st.title("RAG para Revisão Narrativa — múltiplos PDFs ➜ texto coeso com (Label:pag) e muitas citações")

with st.sidebar:
    st.header("Parâmetros")
    chunk_size = st.number_input(
        "Tamanho do chunk (aprox. palavras)",
        min_value=400, max_value=4000, value=int(os.getenv("CHUNK_SIZE", "1200")), step=100
    )
    overlap = st.number_input(
        "Sobreposição", min_value=0, max_value=1000, value=int(os.getenv("CHUNK_OVERLAP", "200")), step=50
    )

    st.markdown("— **Cobertura & Riqueza de citações** —")
    # valores pensados para maximizar evidências usando muitos PDFs
    topk = st.slider("Top-K (busca inicial)", 10, 200, 80, step=10)
    min_per_doc = st.slider("Mínimo por documento", 1, 5, 2, step=1)
    max_per_doc = st.slider("Máximo por documento", 2, 8, 4, step=1)
    max_total = st.slider("Máximo total de chunks no prompt", 8, 40, 18, step=1)
    min_cites_per_paragraph = st.slider("Citações mínimas por parágrafo", 1, 4, 2, step=1)

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
        placeholder="ex.: IA generativa e autenticidade de marca no marketing"
    )
with col2:
    run_btn = st.button("Gerar revisão", type="primary")

# ===== Estado =====
if "vector" not in st.session_state:
    st.session_state.vector = None
if "mapping" not in st.session_state:
    st.session_state.mapping = {}
if "chunks" not in st.session_state:
    st.session_state.chunks: List[Chunk] = []
if "doc_labels" not in st.session_state:
    st.session_state.doc_labels: Dict[str, str] = {}

# ===== Ingestão a partir dos uploads (com labels Autor-Ano) =====
if uploaded and st.session_state.vector is None:
    with st.spinner("Processando PDFs, inferindo labels e criando índice..."):
        vec = VectorIndex()
        mapping: Dict[str, str] = {}
        all_chunks: List[Chunk] = []

        for i, uf in enumerate(uploaded):
            raw = uf.read()
            pages, title = read_pdf(raw)
            filename_stem = os.path.splitext(os.path.basename(title))[0]
            label = build_doc_citation_key(pages, filename_stem)

            doc_id = f"doc{i+1}"
            st.session_state.doc_labels[doc_id] = label
            mapping[filename_stem] = title

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

# ===== Execução =====
if run_btn:
    if not theme.strip():
        st.warning("Informe um tema/pergunta de pesquisa.")
    elif st.session_state.vector is None:
        st.warning("Envie os PDFs antes.")
    else:
        with st.spinner("Buscando e sintetizando com máxima cobertura de fontes..."):
            hits = st.session_state.vector.search(theme, k=int(topk))
            selected = select_for_coverage_and_richness(
                hits,
                min_per_doc=int(min_per_doc),
                max_per_doc=int(max_per_doc),
                max_total=int(max_total)
            )

            # Auditoria opcional dos trechos
            with st.expander("Trechos selecionados (auditoria)"):
                for ch in selected:
                    st.markdown(f"**{build_citation(ch)} — {ch.title}**")
                    st.write(ch.text)
                    st.divider()

            prompt = make_prompt(theme, selected, min_cites_per_paragraph=int(min_cites_per_paragraph))

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

            st.markdown("### Arquivos utilizados (Label → Arquivo)")
            for label, stem in used_files:
                st.write(f"- **{label}** → {stem}")

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
- O texto é narrativo, com **muitas citações** no corpo no formato (Label:pag), priorizando **diversidade de fontes**.
- A lista “Arquivos utilizados (Label → Arquivo)” ajuda você a montar as referências. Sem APA automática.
- Para textos ainda mais ricos, aumente `Top-K`, `Mínimo por documento` e/ou `Máximo total de chunks`.
- Se PDFs forem scaneados sem texto, rode OCR (ex.: Tesseract/ocrmypdf) antes.
"""
)


