import os
import json
import time
import re
import PyPDF2
from google import genai
import faiss
import numpy as np
from google.genai import types

# ---------- Config ----------
INDEX_FILE = "my_index.faiss"
CHUNKS_FILE = "chunks.json"
PDF_FILE = "sample.pdf"
EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL = "gemini-2.5-flash"
TOP_K = 3  # can lower back to 3 now, since chunks are cleaner

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

# ---------- Phase 1: Load PDF ----------
def load_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# ---------- Phase 2: Chunking (section-based, fixes the root cause) ----------
def chunk_by_sections(text):
    sections = re.split(r'\n(?=[A-Z][A-Z\s&]{3,}\n)', text)
    return [s.strip() for s in sections if s.strip()]

# ---------- Phase 3: Embeddings (with retry) ----------
def get_embedding(text, retries=3):
    for attempt in range(retries):
        try:
            result = client.models.embed_content(model=EMBED_MODEL, contents=text)
            return result.embeddings[0].values
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise Exception("Embedding failed after retries")

# ---------- Phase 4: Build or Load Index ----------
def build_index():
    print("No saved index found — building fresh (this calls the API)...")
    doc_text = load_pdf(PDF_FILE)
    chunks = chunk_by_sections(doc_text)

    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\n[Chunk {i}] {c[:80]}...")

    all_embeddings = [get_embedding(c) for c in chunks]

    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_array = np.array(all_embeddings).astype('float32')
    index.add(embeddings_array)

    chunk_lookup = {i: chunks[i] for i in range(len(chunks))}

    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "w") as f:
        json.dump(chunk_lookup, f)

    print(f"\nIndex built and saved: {index.ntotal} vectors.")
    return index, chunk_lookup

def load_index():
    print("Saved index found — loading from disk (no API calls needed).")
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "r") as f:
        raw_lookup = json.load(f)
    chunk_lookup = {int(k): v for k, v in raw_lookup.items()}
    return index, chunk_lookup

def get_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        return load_index()
    else:
        return build_index()

index, chunk_lookup = get_index()

# ---------- Phase 5: Retrieval ----------
def retrieve(question, top_k=TOP_K):
    query_vector = get_embedding(question)
    query_array = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_array, top_k)
    return [chunk_lookup[i] for i in indices[0]]

# ---------- Phase 6: Generation ----------
def generate_answer(question, retrieved_chunks, retries=3):
    print("\n--- RETRIEVED CHUNKS (debug) ---")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk[:100]}...")

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, tools=[])
            )
            return response.text
        except Exception as e:
            print(f"Generation attempt {attempt+1} failed: {e}")
            time.sleep(2)
    raise Exception("Generation failed after retries")

# ---------- Run ----------
if __name__ == "__main__":
    question = "What does Mugilan done in prep room project"
    results = retrieve(question)
    answer = generate_answer(question, results)
    print("\n--- FINAL ANSWER ---")
    print(answer)