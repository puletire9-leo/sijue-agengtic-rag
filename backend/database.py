import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


def init_db() -> None:
    # Delayed imports to avoid circular dependency and register all ORM models with Base.metadata.
    import models  # noqa: F401
    import memory.providers.short_term  # noqa: F401
    import memory.providers.long_term  # noqa: F401

    if os.getenv("DB_RUN_MIGRATIONS", "").lower() in ("1", "true"):
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config(str(Path(__file__).resolve().parent / "alembic.ini"))
        command.upgrade(alembic_cfg, "head")
    else:
        Base.metadata.create_all(bind=engine)
