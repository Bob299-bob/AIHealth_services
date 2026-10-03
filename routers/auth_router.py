from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    admin_only
)

from database import get_db
from models import User
from schemas import RegisterRequest, LoginRequest
from auth import (
    hash_password,
    verify_password,
    create_access_token
)

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)


@router.post("/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db)
):

    existing_user = (
        db.query(User)
        .filter(User.email == data.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    user = User(
        name=data.name,
        email=data.email,
        password_hash=hash_password(data.password),
        role="user"
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return {
        "message": "Registration successful"
    }


@router.post("/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db)
):

    user = (
        db.query(User)
        .filter(User.email == data.email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(
        data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(
        user.id,
        user.role
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user.role
    }
@router.get("/profile")
def profile(
    current_user=Depends(get_current_user)
):

    return {
        "message": "You are authenticated",
        "user": current_user
    }


@router.get("/admin")
def admin_panel(
    current_user=Depends(admin_only)
):

    return {
        "message": "Welcome Admin",
        "user": current_user
    }