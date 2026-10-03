import os
import uuid
import tempfile

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File
)

from auth import get_current_user
from schemas import RAGQuestionRequest

from Services.docBrain import (
    pdf_extract,
    create_rag,
    ask_rag
)


router = APIRouter(
    prefix="/api/rag",
    tags=["RAG"]
)


# Temporary in-memory document storage
documents = {}


# =========================
# UPLOAD DOCUMENT
# =========================

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    temp_path = None

    try:

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        # =========================
        # TEMPORARY PDF FILE
        # =========================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(content)
            temp_path = temp_file.name


        # =========================
        # PDF → TEXT
        # =========================

        pdf_text = pdf_extract(
            temp_path
        )


        # =========================
        # TEXT → CHUNKS
        # → TF-IDF
        # → FAISS
        # =========================

        index, chunks, vectorizer = create_rag(
            pdf_text
        )


        # =========================
        # DOCUMENT ID
        # =========================

        document_id = str(
            uuid.uuid4()
        )


        # =========================
        # STORE DOCUMENT
        # =========================

        documents[document_id] = {

            "index": index,

            "chunks": chunks,

            "vectorizer": vectorizer,

            "filename": file.filename,

            "user_id": current_user["user_id"]

        }


        return {

            "message":
                "Document uploaded successfully",

            "document_id":
                document_id,

            "filename":
                file.filename,

            "chunks":
                len(chunks)

        }


    except HTTPException:
        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Document processing error: {str(e)}"
        )


    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(
                temp_path
            )


# =========================
# ASK QUESTION
# =========================

@router.post("/ask")
def ask_document(
    data: RAGQuestionRequest,
    current_user=Depends(get_current_user)
):

    # =========================
    # FIND DOCUMENT
    # =========================

    document = documents.get(
        data.document_id
    )


    if not document:

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )


    # =========================
    # CHECK OWNERSHIP
    # =========================

    if (
        document["user_id"]
        != current_user["user_id"]
    ):

        raise HTTPException(
            status_code=403,
            detail="You do not have access to this document."
        )


    # =========================
    # VALIDATE QUESTION
    # =========================

    if not data.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    try:

        # =========================
        # ASK RAG
        # =========================

        result = ask_rag(

            data.question,

            document["index"],

            document["chunks"],

            document["vectorizer"]

        )


        return result


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"RAG error: {str(e)}"
        )
