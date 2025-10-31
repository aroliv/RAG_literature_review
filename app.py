# app.py - RAG Streamlit
from __future__ import annotations
import os, io, re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
try:
    import google.generativeai as genai
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False
try:
    from docx import Document
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False
DEFAULT_EMB = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    title: str
def clean_text(s: str) -> str:
    s = re.sub(r'\s+', ' ', s)
    return s.strip()
def read_pdf(path_or_bytes: bytes | str) -> Tuple[List[str], str]:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        doc = fitz.open(stream=path_or_bytes, filetype='pdf')
        title = 'uploaded.pdf'
    else:
        doc = fitz.open(path_or_bytes)
        title = os.path.basename(path_or_bytes)
    pages = [p.get_text('text') for p in doc]
    return pages, title
def chunk_pages(pages: List[str], title: str, doc_id: str, chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
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
    chunks: List[Chunk] = []
    start = 0
    cid = 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        text = clean_text(' '.join(tokens[start:end]))
        p_start = token_index_to_page(start)
        p_end = token_index_to_page(end)
        chunks.append(Chunk(doc_id=doc_id, chunk_id=f'{doc_id}-{cid}', text=text, page_start=p_start, page_end=p_end, title=title))
        cid += 1
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks
class VectorIndex:
    def __init__(self, model_name: str = DEFAULT_EMB):
        self.emb_model = SentenceTransformer(model_name)
        self.index = None
        self.vectors = None
        self.meta: List[Chunk] = []
    def add(self, chunks: List[Chunk]):
        embs = self.emb_model.encode([c.text for c in chunks], convert_to_numpy=True, normalize_embeddings=True)
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])
            self.vectors = embs
        else:
            import numpy as np
            self.vectors = np.vstack([self.vectors, embs])
        self.meta.extend(chunks)
        self.index.add(embs)
    def search(self, query: str, k: int = 20):
        q = self.emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            results.append((self.meta[idx], float(score)))
        return results
def diversify_by_doc(hits, per_doc: int = 3, max_total: int = 12) -> List[Chunk]:
    from collections import defaultdict
    bydoc = defaultdict(list)
    for ch, sc in hits:
        bydoc[ch.doc_id].append((ch, sc))
    for d in bydoc:
        bydoc[d].sort(key=lambda x: x[1], reverse=True)
    out = []
    round_i = 0
    keys = sorted(bydoc.keys())
    while len(out) < max_total:
        added = 0
        for d in keys:
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
    return f'[{base}:{ch.page_start+1}]'
def make_prompt(theme: str, selected: List[Chunk]) -> str:
    context_blocks = [f"{build_citation(ch)}\n{ch.text}" for ch in selected]
    context = "\n\n".join(context_blocks)
    return (
        'Voce e um assistente de revisao sistematica.\n'
        'Responda APENAS com base nos trechos fornecidos; se faltar evidencia, diga nao ha suporte.\n'
        'Use secoes com titulos claros e bullets.\n'
        'Inclua citacoes no formato [doc:pag].\n\n'
        f'TEMA: {theme}\n\n'
        'TRECHOS:\n' + context + '\n\n'
        'GERAR (pt-BR):\n'
        '# 1. Contexto e definicoes\n'
        '# 2. Metodos predominantes\n'
        '# 3. Achados-chave (com citacoes)\n'
        '# 4. Lacunas e controversias\n'
        '# 5. Agenda de pesquisa (3-7 proposicoes)\n'
        '# 6. Limitacoes do corpus\n'
    )
def call_gemini(prompt: str, model_name: str = 'gemini-1.5-flash') -> str:
    resp = genai.GenerativeModel(model_name).generate_content(prompt)
    if hasattr(resp, 'text') and resp.text:
        return resp.text
    return 'Nao foi possivel gerar texto com o Gemini.'
def extractive_review(theme: str, selected: List[Chunk]) -> str:
    lines = [f'## Revisao (extrativa) — {theme}']
    for ch in selected:
        lines.append(f'- {ch.text.strip()} {build_citation(ch)}')
    return '\n'.join(lines)
def to_markdown(theme: str, body: str, mapping: Dict[str, str]) -> str:
    refs = [f'- [{k}] {v}' for k, v in sorted(mapping.items())]
    return f'# Revisao de Literatura — {theme}\n\n' + body + '\n\n## Referencias (arquivo:base)\n' + '\n'.join(refs)
def to_docx(theme: str, body_md: str) -> bytes:
    doc = Document()
    doc.add_heading(f'Revisao de Literatura — {theme}', level=0)
    for line in body_md.splitlines():
        if line.startswith('# '):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith('- '):
            doc.add_paragraph(line[2:].strip())
        else:
            if line.strip() == '':
                doc.add_paragraph('')
            else:
                doc.add_paragraph(line)
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()
st.set_page_config(page_title='RAG Review - PDFs -> Revisao', layout='wide')
st.title('RAG para Revisao Bibliografica — multiplos PDFs -> Revisao estruturada')
with st.sidebar:
    st.header('Parametros')
    chunk_size = st.number_input('Tamanho do chunk (aprox. palavras)', 400, 3000, int(os.getenv('CHUNK_SIZE', '1200')), step=100)
    overlap = st.number_input('Sobreposicao', 0, 800, int(os.getenv('CHUNK_OVERLAP', '200')), step=50)
    topk = st.slider('Top-K inicial (busca)', 5, 100, 30, step=5)
    per_doc = st.slider('Diversidade por documento (max. chunks)', 1, 10, 3, step=1)
    max_total = st.slider('Maximo total de chunks no prompt', 5, 30, 12, step=1)
    st.subheader('Gemini')
    gemini_key = st.text_input('GEMINI_API_KEY', type='password', value=os.getenv('GEMINI_API_KEY', ''))
    gemini_model = st.selectbox('Modelo', ['gemini-1.5-flash', 'gemini-1.5-pro'], index=0)
    if gemini_key and USE_GEMINI:
        genai.configure(api_key=gemini_key)
uploaded = st.file_uploader('Envie multiplos PDFs', type=['pdf'], accept_multiple_files=True)
st.markdown('---')
col1, col2 = st.columns([2,1])
with col1:
    theme = st.text_area('Tema / pergunta de pesquisa', placeholder='ex.: Impacto de IA generativa na percepcao de autenticidade de marcas')
with col2:
    run_btn = st.button('Gerar revisao', type='primary')
if 'vector' not in st.session_state: st.session_state.vector = None
if 'mapping' not in st.session_state: st.session_state.mapping = {}
if 'chunks' not in st.session_state: st.session_state.chunks = []
if uploaded and st.session_state.vector is None:
    with st.spinner('Processando PDFs e criando indice...'):
        vec = VectorIndex()
        mapping = {}
        all_chunks = []
        for i, uf in enumerate(uploaded):
            raw = uf.read()
            pages, title = read_pdf(raw)
            doc_id = f'doc{i+1}'
            mapping[os.path.splitext(title)[0]] = title
            cs = chunk_pages(pages, title=title, doc_id=doc_id, chunk_size=int(chunk_size), overlap=int(overlap))
            all_chunks.extend(cs)
        vec.add(all_chunks)
        st.session_state.vector = vec
        st.session_state.mapping = mapping
        st.session_state.chunks = all_chunks
    st.success(f'Indexados {len(st.session_state.chunks)} chunks de {len(uploaded)} PDFs.')
if run_btn:
    if not theme.strip():
        st.warning('Informe um tema/pergunta de pesquisa.')
    elif st.session_state.vector is None:
        st.warning('Envie os PDFs antes.')
    else:
        with st.spinner('Buscando e sintetizando...'):
            hits = st.session_state.vector.search(theme, k=int(topk))
            selected = diversify_by_doc(hits, per_doc=int(per_doc), max_total=int(max_total))
            with st.expander('Trechos selecionados (para auditoria)'):
                for ch in selected:
                    st.markdown(f'**{build_citation(ch)} — {ch.title}**')
                    st.write(ch.text)
                    st.divider()
            if gemini_key and USE_GEMINI:
                prompt = make_prompt(theme, selected)
                body_text = call_gemini(prompt, model_name=gemini_model)
            else:
                body_text = extractive_review(theme, selected)
            md = to_markdown(theme, body_text, st.session_state.mapping)
            st.markdown('### Revisao gerada')
            st.write(body_text)
            st.markdown('---')
            st.subheader('Exportar')
            st.download_button('Baixar Markdown (.md)', data=md.encode('utf-8'), file_name='revisao.md', mime='text/markdown')
            if HAS_DOCX:
                docx_bytes = to_docx(theme, body_text)
                st.download_button('Baixar Word (.docx)', data=docx_bytes, file_name='revisao.docx', mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
            else:
                st.caption('python-docx nao disponivel — instale para exportar .docx')
st.markdown('---\nDicas\n- Se o PDF for escaneado sem texto, faca OCR antes (ex.: Tesseract).\n- Ajuste Top-K e Diversidade por documento para controlar foco e abrangencia.\n- Para maior qualidade, troque o embedding para all-MiniLM-L12-v2.\n- O app mantem tudo em memoria; use ingest.py para indice FAISS em disco.')