from src.api.schemas import GenerateApiKeyRequest, TokenResponse
from src.database.session import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, HTTPException
from src.api.middleware.auth import verify_token
from fastapi import APIRouter
from src.services.token_service import TokenService
from typing import List

async def get_token_service(db: AsyncSession = Depends(get_db)):
    return TokenService(db)

def get_system_router():
    router = APIRouter(tags=["System"])

    @router.get("/health")
    async def health_check():
        return {"status": "ok", "message": "API is running"}

    @router.post("/generate-api-key", dependencies=[Depends(verify_token)])
    async def generate_api_key(
        request: GenerateApiKeyRequest,
        token_service: TokenService = Depends(get_token_service),
    ):
        token = await token_service.create_token(request)

        return {"status": "success", "message": "API key generated successfully", "token": token.token}

    @router.get("/tokens", response_model=List[TokenResponse], dependencies=[Depends(verify_token)])
    async def list_tokens(token_service: TokenService = Depends(get_token_service)):
        return await token_service.list_tokens()

    @router.patch("/tokens/{token_id}/toggle", dependencies=[Depends(verify_token)])
    async def toggle_token(
        token_id: int,
        is_active: bool,
        token_service: TokenService = Depends(get_token_service)
    ):
        token = await token_service.toggle_token(token_id, is_active)
        if not token:
            raise HTTPException(status_code=404, detail="Token not found")
        return {"status": "success", "message": f"Token {'activated' if is_active else 'deactivated'}"}

    @router.delete("/tokens/{token_id}", dependencies=[Depends(verify_token)])
    async def delete_token(
        token_id: int,
        token_service: TokenService = Depends(get_token_service)
    ):
        token = await token_service.delete_token(token_id)
        if not token:
            raise HTTPException(status_code=404, detail="Token not found")
        return {"status": "success", "message": "Token deleted successfully"}

    return router