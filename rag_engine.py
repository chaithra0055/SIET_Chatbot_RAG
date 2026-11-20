import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. LOAD PDF
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError("No readable text found in PDF")

    return full_text


# 2. CREATE CHUNKS
def create_chunks(full_text, chunk_size=500, overlap=100):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# 3. BUILD VECTORS
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):
    if not chunks:
        raise ValueError("Cannot create vectors without chunks")

    embed_model = SentenceTransformer(model_name)

    vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index, embed_model


# 4. LOAD BASE
def load_base(pdf_path="01_College_Information.pdf"):
    full_text = load_pdf(pdf_path)
    chunks = create_chunks(full_text)
    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# 5. RETRIEVE CONTENT
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):
    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_vector = embed_model.encode(
        [question],
        convert_to_numpy=True
    ).astype(np.float32)

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# 6. ASK RAG
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):
    if not api_key:
        raise ValueError("API key missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks=num_chunks
    )

    prompt = f"""
You are a professional AI assistant for Shridevi Institute of Engineering and Technology.

Answer the user's question using ONLY the provided document background.

Rules:
- Give correct and clear answers.
- Do not invent facts.
- Do not use outside knowledge.
- If the answer is missing from the document, say "Information Missing".
- Keep the answer simple and relevant.

Document Background:
{background}

User Question:
{question}

Answer:
"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content rag.engine
