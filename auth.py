from passlib.context import CryptContext
from jose import jwt

SECRET_KEY = "my-secret-key-change-later"
ALGORITHM = "HS256"

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(password, hashed_password):
    return pwd_context.verify(password, hashed_password)


def create_access_token(user_id, role):
    payload = {
        "user_id": user_id,
        "role": role
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


# -------------------------
# JWT Verification
# -------------------------

from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login"
)


def get_current_user(
    token: str = Depends(oauth2_scheme)
):
    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        return payload

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )


def admin_only(
    current_user=Depends(get_current_user)
):

    if current_user.get("role") != "admin":

        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user