
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from models import Base, ViolationType


# SQLite database file in the project directory
DATABASE_PATH = Path(__file__).resolve().parent / "traffic_violation.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)


# Enable SQLite foreign-key constraints
@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


DEFAULT_VIOLATION_TYPES = [
    {
        "code": "NO_HELMET",
        "name": "Riding Without Helmet",
        "description": "Riding a two-wheeler without a helmet.",
    },
    {
        "code": "RED_LIGHT",
        "name": "Red-Light Violation",
        "description": "Crossing the stop line while the signal is red.",
    },
    {
        "code": "WRONG_LANE",
        "name": "Wrong-Lane Driving",
        "description": "Driving in a prohibited lane or direction.",
    },
    {
        "code": "OVERLOADING",
        "name": "Excess Passengers on Two-Wheeler",
        "description": "Carrying more passengers than permitted.",
    },
]


def init_db():
    """Create tables and seed default violation categories."""
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        for item in DEFAULT_VIOLATION_TYPES:
            existing = session.scalar(
                select(ViolationType).where(
                    ViolationType.code == item["code"]
                )
            )

            if existing is None:
                session.add(ViolationType(**item))

        session.commit()


def get_db():
    """Provide a database session for application code."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print(f"SQLite database initialized at: {DATABASE_PATH}")
