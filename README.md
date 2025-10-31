# RAG para Revisão Bibliográfica — Streamlit + FAISS + Gemini

App para gerar **revisões estruturadas** a partir de **múltiplos PDFs** usando RAG e **Gemini**.  
Você faz upload dos PDFs, informa o **tema/pergunta de pesquisa**, e o app busca trechos relevantes, **diversifica por documento** e gera uma revisão com **citações** no formato `[arquivo:página]`.  
Também exporta em **Markdown** e **Word (.docx)**.

---

## 🧠 Recursos

- Upload de múltiplos PDFs (via **PyMuPDF**)
- Chunking configurável (**tamanho / overlap**)
- Busca semântica (**SentenceTransformers** + **FAISS**)
- **Diversificação por documento** (evita viés em um único paper)
- **Síntese com Gemini** (`google-generativeai`) ou modo **extrativo** (sem LLM)
- Exportação em **Markdown** e **.docx**
- (Opcional) Ingestão/persistência de índice local via `ingest.py`

---

## ⚙️ Instalação rápida

```bash
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> 💡 **Dica:** em algumas distros, `faiss-cpu` funciona melhor com **Python 3.10+**.  
> Alternativa:  
> ```bash
> conda install -c conda-forge faiss-cpu
> ```

---

## 🔐 Variáveis de ambiente

Crie um arquivo `.env` (ou preencha no sidebar do app):

```bash
GEMINI_API_KEY="sua_chave"
GEMINI_MODEL="gemini-1.5-flash"  # ou gemini-1.5-pro
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

---


## 🚀 Como rodar

```bash
streamlit run app.py
```

1. Envie os PDFs
2. Escreva o tema ou pergunta de pesquisa
3. Clique “Gerar revisão”
---


## 💾 Persistência (opcional)

Coloque seus PDFs em data/papers/ e gere um índice FAISS persistente:
```
python ingest.py  # cria/atualiza data/index/
```

Isso permite reaproveitar embeddings entre execuções.

---

## 📤 Exportação

- Markdown: botão “Baixar Markdown (.md)” gera revisao.md
- Word (.docx): requer python-docx (já incluso em requirements.txt)
