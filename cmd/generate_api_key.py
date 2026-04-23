import asyncio
import os
import sys
import secrets

# Menambahkan root project ke sys.path agar bisa import dari src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database.session import init_db, get_sessionmaker
from src.database.entity.token import Token

async def generate_admin_token(name: str):
    """
    Script to generate an initial admin API key.
    Usage: python cmd/generate_api_key.py [name]
    """
    # 1. Pastikan database dan tabel sudah terbuat
    await init_db()
    
    # 2. Buat session database
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        # Generate token random
        token_str = secrets.token_hex(16)
        
        # Buat entity token baru sebagai admin
        new_token = Token(
            name=name,
            token=token_str,
            is_active=True,
            is_admin=True
        )

        session.add(new_token)
        await session.commit()

        print("\n" + "="*50)
        print(f" ADMIN API KEY GENERATED SUCCESSFULLY")
        print("="*50)
        print(f" Name  : {name}")
        print(f" Token : {token_str}")
        print("="*50)

if __name__ == "__main__":
    # Ambil nama dari argumen command line (default: root)
    admin_name = sys.argv[1] if len(sys.argv) > 1 else "root"
    asyncio.run(generate_admin_token(admin_name))
