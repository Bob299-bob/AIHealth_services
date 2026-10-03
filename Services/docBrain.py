import os
import faiss
import pdfplumber

from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


def pdf_extract(data_path):

    pdf_text = ""

    with pdfplumber.open(data_path) as file:

        for page in file.pages:

            page_text = page.extract_text()

            if page_text:
                pdf_text += page_text + "\n"

    return pdf_text


def create_rag(pdf_text):

    if not pdf_text.strip():
        raise ValueError(
            "No readable text found in PDF."
        )

    chunks = splitter.split_text(pdf_text)

    embeddings = embedding_model.encode(
        chunks
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, chunks


def retrieve(query, index, chunks, k=5):

    if not chunks:
        return []

    query_embedding = embedding_model.encode(
        [query]
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(
        query_embedding,
        k=min(k, len(chunks))
    )

    retrieved_chunks = []

    for idx in indices[0]:

        if idx != -1:
            retrieved_chunks.append(
                chunks[idx]
            )

    return retrieved_chunks


def generate_answer(query, retrieved_chunks):

    if not retrieved_chunks:
        return (
            "I could not find relevant information "
            "in the uploaded document."
        )

    context = "\n\n".join(
        retrieved_chunks
    )

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question using ONLY
the information provided in the context.

If the answer is not present in the context,
say:

"I could not find this information in the document."

Do not invent facts.

Keep the answer clear and simple.

Context:
----------------
{context}
----------------

User Question:
{query}

Answer:
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions using "
                    "provided document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content


def ask_rag(query, index, chunks):

    retrieved_chunks = retrieve(
        query,
        index,
        chunks,
        k=5
    )

    answer = generate_answer(
        query,
        retrieved_chunks
    )

    return {
        "question": query,
        "answer": answer,
        "sources": retrieved_chunks
    }