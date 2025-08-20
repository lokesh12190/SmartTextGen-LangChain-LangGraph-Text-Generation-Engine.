# SmartTextGen — LangChain + LangGraph Text Generation Engine (with optional RAG & citations)

SmartTextGen is a practical, CPU-friendly text generation stack. It orchestrates prompts and retrieval with **LangChain + LangGraph**, runs **Hugging Face** models locally, and can ground answers in your PDFs using **Chroma**. A built-in **Answer → Critic → Revise** loop improves faithfulness and encourages citations.

> Works entirely offline with local models. Great for research assistants, knowledge bots, and structured content pipelines.

---

## ✨ Features
- **RAG over your PDFs** (optional): ingest, chunk, embed, and retrieve from `data/docs/`
- **Agentic loop (LangGraph)**: Answer → Critic → (optional) Revise with stop conditions
- **Citations**: Answers can include `[docN]` references to retrieved chunks
- **Local models**: Uses Hugging Face `flan-t5` for generation, `all-MiniLM-L6-v2` for embeddings
- **CPU-friendly defaults**: Sensible truncation and context caps to avoid token overflows
- **Zero API keys required**

---

## 🧱 Tech Stack
- **LangChain**, **LangGraph**
- **Hugging Face** `transformers`, **sentence-transformers**
- **ChromaDB** vector store
- **PyPDF** for PDF parsing

---

## 🚀 Quickstart (Windows + conda)

```powershell
# 1) Create & activate env
conda create -n research-assistant python=3.10 -y
conda activate research-assistant

# 2) Install deps
pip install -r requirements.txt

# 3) Add at least one PDF to:
#    data/docs/ (e.g., paper1.pdf)

# 4) Run the CLI (from the project root)
python -m src.cli
