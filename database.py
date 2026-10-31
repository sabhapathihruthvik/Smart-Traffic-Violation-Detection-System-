
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker


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
        "fine_amount": 500,
    },
    {
        "code": "RED_LIGHT",
        "name": "Red-Light Violation",
        "description": "Crossing the stop line while the signal is red.",
        "fine_amount": 1000,
    },
    {
        "code": "WRONG_LANE",
        "name": "Wrong-Lane Driving",
        "description": "Driving in a prohibited lane or direction.",
        "fine_amount": 2000,
    }
]


def init_db(target_engine: Engine = engine) -> None:
    """Add and populate fine amounts once, without changing other records."""
    with target_engine.begin() as connection:
        if "violation_types" not in inspect(connection).get_table_names():
            return

        columns = {
            column["name"]
            for column in inspect(connection).get_columns("violation_types")
        }
        if "fine_amount" in columns:
            return

        connection.execute(
            text(
                "ALTER TABLE violation_types "
                "ADD COLUMN fine_amount INTEGER NOT NULL DEFAULT 0"
            )
        )
        for item in DEFAULT_VIOLATION_TYPES:
            connection.execute(
                text(
                    "UPDATE violation_types SET fine_amount = :fine_amount "
                    "WHERE code = :code"
                ),
                {
                    "fine_amount": item["fine_amount"],
                    "code": item["code"],
                },
            )


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
