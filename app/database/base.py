from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Create async engine with MySQL optimizations
engine = create_async_engine(
  settings.DATABASE_URL,
  echo=settings.DEBUG,
  future=True,
  # MySQL-specific optimizations for better concurrency
  pool_size=10,  # More connections for MySQL
  max_overflow=20,  # Allow more overflow connections
  pool_pre_ping=True,  # Verify connections before use
  pool_recycle=3600,  # Recycle connections after 1 hour
  # MySQL handles concurrent connections much better than SQLite
)

# Create session factory
async_session_maker = async_sessionmaker(
  engine,
  class_=AsyncSession,
  expire_on_commit=False,  # Don't expire objects after commit
  autoflush=False,  # Manual flush control
  autocommit=False,
)

# Base class for models
Base = declarative_base()


async def get_db():
  """
  Dependency for getting database session
  MySQL version - better concurrent handling
  """
  async with async_session_maker() as session:
    try:
      yield session
      await session.commit()
    except Exception as e:
      await session.rollback()
      logger.error(f"Database error: {e}")
      raise
    finally:
      await session.close()


async def init_db():
  """
  Initialize database tables for MySQL
  """
  logger.info("🔄 Initializing MySQL database...")

  async with engine.begin() as conn:
    await conn.run_sync(Base.metadata.create_all)
    logger.info("✅ MySQL database initialized successfully")
    logger.info(f"✅ Connected to: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'MySQL'}")