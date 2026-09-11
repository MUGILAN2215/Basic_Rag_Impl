# RagSystem — RAG (Retrieval-Augmented Generation) Q&A Bot

A document Q&A system built from scratch in raw Python (no LangChain) to understand the RAG pipeline at a fundamental level, then optionally wrapped with LangChain.

---

## What This Project Does

Given a PDF document, this system lets you ask natural-language questions and get answers grounded **only** in that document's content — instead of relying on an LLM's general training knowledge (which can hallucinate).

**Pipeline:**
```
PDF → Extract Text → Chunk → Embed → Store (Vector DB) → Retrieve → Generate Answer
```

---

## Problem This Solves

A plain LLM (Gemini, GPT, etc.) only knows what it was trained on. Ask it about a specific document — a resume, a company policy, internal notes — and it either:
- Says "I don't know," or
- **Hallucinates** — makes up a plausible-sounding but wrong answer

RAG fixes this by **retrieving relevant facts first**, then instructing the LLM to answer using only those facts.

---

## Phase-by-Phase Breakdown

### Phase 1: Load & Extract Text
**Tool used:** `PyPDF2`
**What it does:** PDFs store layout, fonts, and formatting — not plain text. `PyPDF2` opens the file and extracts just the readable text, page by page.

**Alternatives:**
| Tool | When to use instead |
|---|---|
| `pdfplumber` | Better table/layout extraction |
| `PyMuPDF (fitz)` | Faster, more accurate text extraction |
| OCR (`pytesseract`, AWS Textract, Azure Document Intelligence) | Scanned/image-based PDFs with no embedded text layer |
| `unstructured.io` | Mixed documents with tables, images, and text needing structure-aware parsing |

---

### Phase 2: Chunking
**Method used:** Manual fixed-size chunking (500 characters, 50-character overlap)
**What it does:** Splits the full document into smaller pieces. Smaller chunks improve retrieval precision — you fetch just the relevant paragraph, not the whole document. Overlap prevents cutting a sentence/idea in half at a chunk boundary.

**Alternatives:**
| Method | Why you'd use it |
|---|---|
| `RecursiveCharacterTextSplitter` (LangChain) | Splits on paragraph/sentence boundaries first — avoids cutting mid-sentence |
| Token-based chunking (`tiktoken`) | Chunks by LLM token count instead of characters — more accurate against model limits |
| Semantic chunking | Splits where the *meaning* changes, using an embedding model to detect topic shifts |
| Structure-aware chunking | Splits on document headings (e.g., "WORK EXPERIENCE", "EDUCATION") — ideal for structured docs like resumes |

**Known limitation found in this project:** Fixed-size chunking cut across resume sections (e.g., mixed "Skills" and "Objective" in one chunk), which hurt retrieval accuracy. Structure-aware chunking would fix this.

---

### Phase 3: Embeddings
**Model used:** `gemini-embedding-001` (via the `google-genai` SDK)
**What it does:** Converts each text chunk into a 3072-number vector representing its meaning. Chunks with similar meaning end up mathematically close together in vector space.

**Why this model:**
- Current, actively supported Gemini embedding model (replaces the deprecated `embedding-001` and soon-to-be-retired `text-embedding-004`)
- Free tier available via Google AI Studio
- Strong multilingual support (100+ languages), useful if documents aren't English-only
- Supports Matryoshka Representation Learning (MRL) — output dimensions can be scaled down (e.g., 768 instead of 3072) to save storage, without re-embedding

**Alternatives:**
| Model | Why you'd pick it |
|---|---|
| `text-embedding-004` (Google, legacy) | Being retired — avoid for new projects |
| OpenAI `text-embedding-3-small` / `text-embedding-3-large` | Popular alternative, similar quality, different pricing |
| `sentence-transformers` (open-source, e.g., `all-MiniLM-L6-v2`) | Free, runs locally, no API calls/costs — good for offline or privacy-sensitive use |
| Cohere Embed | Strong for retrieval-specific tasks, has multilingual variants |

---

### Phase 4: Vector Store
**Tool used:** FAISS (`faiss-cpu`), `IndexFlatL2`
**What it does:** Stores all chunk embeddings in memory and enables fast similarity search — "find the vectors closest in meaning to this query vector."

**Why this tool:**
- Free, open-source (built by Meta)
- No server/database setup required — runs in-process
- Fast enough for small-to-medium datasets
- Industry-standard for learning/prototyping RAG systems

**Limitation:** Currently in-memory only — data is lost when the script ends. Production systems persist the index to disk (`faiss.write_index`) or use a managed vector database.

**Alternatives:**
| Tool | Why you'd pick it |
|---|---|
| Pinecone | Managed cloud vector DB, no infra to maintain, scales easily |
| Weaviate | Open-source, supports hybrid search (keyword + vector) |
| Chroma | Lightweight, popular for local RAG prototyping, easy persistence |
| Milvus | Built for large-scale production vector search |
| pgvector (Postgres extension) | If you already use Postgres — adds vector search without a new database |

---

### Phase 5: Retrieval
**What it does:** Converts the user's question into a vector (same embedding model), then asks FAISS for the `top_k` most similar stored chunks by distance.

**Key parameter:** `top_k=3` — controls how many chunks are retrieved. Too low risks missing context; too high adds noise and cost.

**Production improvement not included here:** Re-ranking — retrieve a larger initial set (e.g., top 10) with FAISS, then use a smarter model to re-rank and narrow down to the true best matches. Improves accuracy at the cost of extra latency.

---

### Phase 6: Generation
**Model used:** `gemini-2.5-flash`
**What it does:** Takes the retrieved chunks + the original question, and generates a final answer — instructed to use *only* the provided context (grounding), reducing hallucination.

**Why this model:**
- Fast and low-cost, well-suited for short, context-grounded answers
- Good balance of speed vs. quality for a Q&A use case

**Alternatives:**
| Model | Why you'd pick it |
|---|---|
| `gemini-2.5-pro` | Higher quality reasoning, for more complex questions — slower/costlier |
| GPT-4o / GPT-4o-mini (OpenAI) | Common alternative, similar RAG use case |
| Claude (Anthropic) | Strong at following grounding instructions precisely, good for compliance-sensitive answers |
| Open-source (Llama 3, Mistral, via Ollama) | Free, runs locally — no API costs, but needs local compute |

---

## Known Limitations (Current Version)

- **No persistence** — FAISS index and embeddings are rebuilt from scratch on every run (re-embeds all chunks, repeated API calls)
- **Naive chunking** — fixed character-size splitting cuts across logical sections
- **No re-ranking** — retrieval relies purely on raw vector distance
- **No OCR support** — scanned/image-only PDFs would return no text
- **Single document only** — no multi-document indexing yet

## Possible Next Steps

- Persist FAISS index to disk to avoid re-embedding on every run
- Switch to structure-aware or recursive chunking
- Add LangChain wrapper to compare abstraction vs. raw implementation
- Add a Streamlit UI for live demo
- Add retry/backoff logic for API calls (handles transient errors like `503 UNAVAILABLE`)

---

## Tech Stack Summary

| Component | Tool |
|---|---|
| PDF text extraction | PyPDF2 |
| Chunking | Manual fixed-size (Python) |
| Embeddings | Gemini `gemini-embedding-001` |
| Vector store | FAISS (`IndexFlatL2`) |
| Generation | Gemini `gemini-2.5-flash` |
| SDK | `google-genai` |
