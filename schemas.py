from pydantic import BaseModel

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RAGQuestionRequest(BaseModel):
    document_id: str
    question: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str
class DiseaseInput(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float
class ChatRequest(BaseModel):
    message: str