"""Task 5: database trail. Every prediction is stored as one row.

DATABASE_URL decides the database: SQLite locally, PostgreSQL in Docker Compose.
"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    filename: Mapped[str] = mapped_column(String(255))
    image_sha256: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    p_real: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(20))
    latency_ms: Mapped[float] = mapped_column(Float)


class Database:
    def __init__(self, url):
        if url.startswith("sqlite:///"):
            Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, pool_pre_ping=True)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    def add(self, **fields):
        with self.Session() as session:
            row = Prediction(**fields)
            session.add(row)
            session.commit()
            return row

    def get(self, prediction_id):
        with self.Session() as session:
            return session.get(Prediction, prediction_id)

    def latest(self, limit=100):
        with self.Session() as session:
            stmt = select(Prediction).order_by(Prediction.created_at.desc()).limit(limit)
            return list(session.scalars(stmt))

    def is_up(self):
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
