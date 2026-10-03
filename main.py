from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.image_router import router as image_router
from database import engine, Base
from models import User
from routers.auth_router import router as auth_router
from routers.ml_router import router as ml_router
from routers.chat_router import router as chat_router
from routers.rag import router as rag_router


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AI Healthcare Intelligence API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173","https://aihealthcare04.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(ml_router)
app.include_router(image_router)
app.include_router(chat_router)
app.include_router(rag_router)

@app.get("/")
def home():
    return {
        "message": "AI Healthcare API is running"
    }
