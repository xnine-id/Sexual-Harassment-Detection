import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker
from src.database.entity.base import Base
# Import all entities here to ensure they are registered with Base.metadata
from src.database.entity.camera import Camera
from src.database.entity.token import Token

# URL default untuk SQLite (async)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./storage/private/database.db")

# Private variables untuk caching
_engine: AsyncEngine = None
_sessionmaker = None

def get_engine(url: str = DATABASE_URL) -> AsyncEngine:
    """Lazy load engine: hanya dibuat ketika pertama dipanggil."""
    global _engine
    if _engine is None:
        # Untuk SQLite, kita perlu set check_same_thread=False jika menggunakan multiple threads
        connect_args = {"check_same_thread": False} if "sqlite" in url else {}
        _engine = create_async_engine(
            url, 
            echo=True,
            connect_args=connect_args
        )
    return _engine

def get_sessionmaker():
    """Lazy load sessionmaker."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _sessionmaker

# Dependency FastAPI
async def get_db():
    """Dependency FastAPI untuk mendapatkan session database."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        yield session

async def init_db():
    """Fungsi untuk inisialisasi database: membuat file sqlite dan semua tabel."""
    engine = get_engine()
    async with engine.begin() as conn:
        # run_sync digunakan untuk menjalankan operasi synchronous (create_all) di dalam async
        await conn.run_sync(Base.metadata.create_all)
    print(f"Database initialized at {DATABASE_URL}")
