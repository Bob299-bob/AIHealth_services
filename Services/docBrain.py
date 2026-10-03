import os
import faiss
import pdfplumber
import numpy as np

from dotenv import load_dotenv
from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sklearn.feature_extraction.text import TfidfVectorizer


# =========================
# ENVIRONMENT
# =========================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# =========================
# TEXT SPLITTER
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# =========================
# PDF TEXT EXTRACTION
# =========================

def pdf_extract(data_path):

    pdf_text = ""

    with pdfplumber.open(data_path) as file:

        for page in file.pages:

            page_text = page.extract_text()

            if page_text:
                pdf_text += page_text + "\n"

    return pdf_text


# =========================
# CREATE RAG
# =========================

def create_rag(pdf_text):

    if not pdf_text.strip():

        raise ValueError(
            "No readable text found in PDF."
        )


    # Split PDF into chunks
    chunks = splitter.split_text(
        pdf_text
    )


    if not chunks:

        raise ValueError(
            "No text chunks created from PDF."
        )


    # =========================
    # TF-IDF EMBEDDINGS
    # =========================

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )


    embeddings = vectorizer.fit_transform(
        chunks
    )


    # Convert sparse matrix to float32
    embeddings = embeddings.toarray().astype(
        "float32"
    )


    # Normalize vectors
    faiss.normalize_L2(
        embeddings
    )


    # =========================
    # FAISS INDEX
    # =========================

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )


    index.add(
        embeddings
    )


    return (
        index,
        chunks,
        vectorizer
    )


# =========================
# RETRIEVE RELEVANT CHUNKS
# =========================

def retrieve(
    query,
    index,
    chunks,
    vectorizer,
    k=5
):

    if not chunks:

        return []


    # Convert question into TF-IDF vector
    query_embedding = vectorizer.transform(
        [query]
    )


    # Convert to float32
    query_embedding = query_embedding.toarray().astype(
        "float32"
    )


    # Normalize
    faiss.normalize_L2(
        query_embedding
    )


    # Search FAISS
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


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(
    query,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "I could not find relevant information "
            "in the uploaded document."
        )


    # Combine retrieved chunks
    context = "\n\n".join(
        retrieved_chunks
    )


    # =========================
    # PROMPT
    # =========================

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


    # =========================
    # GROQ
    # =========================

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


# =========================
# ASK RAG
# =========================

def ask_rag(
    query,
    index,
    chunks,
    vectorizer
):

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(
        query,
        index,
        chunks,
        vectorizer,
        k=5
    )


    # Generate answer
    answer = generate_answer(
        query,
        retrieved_chunks
    )


    return {

        "question": query,

        "answer": answer,

        "sources": retrieved_chunks
    }
