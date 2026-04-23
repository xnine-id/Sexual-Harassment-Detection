from sqlalchemy.future import select
from src.api.schemas import GenerateApiKeyRequest
from sqlalchemy.ext.asyncio.session import AsyncSession
from src.database.entity.token import Token
import secrets

class TokenService:
    def __init__(self, session: AsyncSession):
        self.session = session

    def _generate_api_key(self):
        return secrets.token_hex(16)

    async def create_token(self, data: GenerateApiKeyRequest, is_admin: bool = False):
        new_api_key = self._generate_api_key()
        new_token = Token(
            name=data.name,
            token=new_api_key,
            expires_at=data.expires_at,
            is_active=True,
            is_admin=is_admin
        )
        self.session.add(new_token)
        await self.session.commit()
        await self.session.refresh(new_token)
        return new_token

    async def list_tokens(self):
        tokens = await self.session.execute(select(Token))
        return tokens.scalars().all()

    async def toggle_token(self, id_token: int, is_active: bool):
        token = await self.session.execute(select(Token).where(Token.id == id_token))
        token = token.scalar_one_or_none()
        if not token:
            return None
        token.is_active = is_active
        await self.session.commit()
        await self.session.refresh(token)
        return token

    async def delete_token(self, id_token: int):
        token = await self.session.execute(select(Token).where(Token.id == id_token))
        token = token.scalar_one_or_none()
        if not token:
            return None
        await self.session.delete(token)
        await self.session.commit()
        return token

    async def get_token(self, id_token: int):
        token = await self.session.execute(select(Token).where(Token.id == id_token))
        token = token.scalar_one_or_none()
        return token