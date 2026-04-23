# src/api/auth.py
from fastapi import Header, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from src.database.session import get_db
from src.database.entity.token import Token

async def verify_token(
    x_api_key: str = Header(None, alias="X-API-Key"),
    authorization: str = Header(None),
    db: AsyncSession = Depends(get_db)
):
    token_str = x_api_key

    # Fallback ke Bearer Token
    if not token_str and authorization and authorization.startswith("Bearer "):
        token_str = authorization.split(" ")[1]

    if not token_str:
        raise HTTPException(status_code=401, detail="Missing API Token")

    result = await db.execute(
        select(Token).where(Token.token == token_str, Token.is_active == True)
    )
    token_obj = result.scalar_one_or_none()

    if not token_obj:
        raise HTTPException(status_code=401, detail="Invalid API Token")

    if token_obj.expires_at and token_obj.expires_at < datetime.now():
        raise HTTPException(status_code=401, detail="Token expired")

    # Update last_used
    token_obj.last_used = datetime.now()
    db.commit()

    return token_obj
