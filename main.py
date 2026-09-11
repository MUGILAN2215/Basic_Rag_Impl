import PyPDF2
from google import genai
import faiss
import numpy as np

# ---------- Phase 1: Load PDF ----------
def load_pdf(path):
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

doc_text = load_pdf("sample.pdf")

# ---------- Phase 2: Chunking ----------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(doc_text)
print(f"Total chunks created: {len(chunks)}")

# ---------- Phase 3: Embeddings ----------
client = genai.Client(api_key=" ")  # paste your real key

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

all_embeddings = []
for chunk in chunks:
    vec = get_embedding(chunk)
    all_embeddings.append(vec)

print(f"Total embeddings created: {len(all_embeddings)}")
print(f"Each vector length: {len(all_embeddings[0])}")

# ---------- Phase 4: Vector Store (FAISS) ----------
dimension = len(all_embeddings[0])
index = faiss.IndexFlatL2(dimension)

embeddings_array = np.array(all_embeddings).astype('float32')
index.add(embeddings_array)

print(f"Total vectors stored in FAISS: {index.ntotal}")

# mapping: FAISS position -> original text chunk
chunk_lookup = {i: chunks[i] for i in range(len(chunks))}

# ---------- Phase 5: Retrieval ----------
def get_query_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def retrieve(question, top_k=3):
    query_vector = get_query_embedding(question)
    query_array = np.array([query_vector]).astype('float32')

    distances, indices = index.search(query_array, top_k)

    retrieved_chunks = [chunk_lookup[i] for i in indices[0]]
    return retrieved_chunks

# test it
question = "What are tools did he Know to use?"
results = retrieve(question)

for i, chunk in enumerate(results):
    print(f"\n--- Match {i+1} ---")
    print(chunk)

# ---------- Phase 6: Generation ----------
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""Answer the question using ONLY the context below. 
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# test it
answer = generate_answer(question, results)
print("\n--- FINAL ANSWER ---")
print(answer)