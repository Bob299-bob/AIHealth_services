import os

from fastapi import APIRouter, Depends, HTTPException
from groq import Groq
from dotenv import load_dotenv

from auth import get_current_user
from schemas import ChatRequest

load_dotenv()

router = APIRouter(
    prefix="/api",
    tags=["AI Assistant"]
)

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing")

client = Groq(api_key=api_key)


SYSTEM_PROMPT = """
You are an AI Healthcare Assistant.

Your job is to provide simple, clear and helpful healthcare
information.

Rules:
1. Use simple English.
2. Explain medical terms in an easy way.
3. Do not claim to provide a medical diagnosis.
4. Do not prescribe medicines or dosages.
5. If symptoms appear serious or emergency-related,
   advise the user to contact a qualified healthcare
   professional or emergency service.
6. Keep answers concise unless the user asks for details.
7. Be polite and supportive.
"""


@router.post("/chat")
def chat(
    data: ChatRequest,
    current_user=Depends(get_current_user)
):
    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": data.message
                }
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content

        return {
            "message": data.message,
            "response": answer
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"AI Assistant error: {str(e)}"
        )