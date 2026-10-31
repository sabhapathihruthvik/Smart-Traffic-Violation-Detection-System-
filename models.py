
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
    Integer,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_number: Mapped[str] = mapped_column(
        String(20), unique=True, index=True, nullable=False
    )
    vehicle_type: Mapped[str] = mapped_column(
        String(30), nullable=False, default="Unknown"
    )
    vehicle_owner: Mapped[str | None] = mapped_column(String(120))
    registered_mobile_number: Mapped[str | None] = mapped_column(String(20))

    violations: Mapped[list["Violation"]] = relationship(
        back_populates="vehicle"
    )


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_name: Mapped[str] = mapped_column(String(100), nullable=False)
    location: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(
        String(20), default="CCTV", nullable=False
    )
    source_uri: Mapped[str | None] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    detection_logs: Mapped[list["DetectionLog"]] = relationship(
        back_populates="camera"
    )
    violations: Mapped[list["Violation"]] = relationship(
        back_populates="camera"
    )


class ViolationType(Base):
    __tablename__ = "violation_types"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(
        String(40), unique=True, nullable=False
    )
    name: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text)
    fine_amount: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    violations: Mapped[list["Violation"]] = relationship(
        back_populates="violation_type"
    )


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(primary_key=True)
    vehicle_id: Mapped[int | None] = mapped_column(
        ForeignKey("vehicles.id"), index=True
    )
    violation_type_id: Mapped[int] = mapped_column(
        ForeignKey("violation_types.id"), nullable=False, index=True
    )
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id"), index=True
    )
    violation_datetime: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    detected_vehicle_number: Mapped[str | None] = mapped_column(String(30))
    confidence: Mapped[float | None] = mapped_column(Float)
    verification_status: Mapped[str] = mapped_column(
        String(20), default="Pending", nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text)

    vehicle: Mapped[Vehicle | None] = relationship(
        back_populates="violations"
    )
    violation_type: Mapped[ViolationType] = relationship(
        back_populates="violations"
    )
    camera: Mapped[Camera | None] = relationship(
        back_populates="violations"
    )
    evidence: Mapped[list["Evidence"]] = relationship(
        back_populates="violation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    violation_id: Mapped[int] = mapped_column(
        ForeignKey("violations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    evidence_type: Mapped[str] = mapped_column(
        String(20), default="Image", nullable=False
    )
    captured_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    description: Mapped[str | None] = mapped_column(Text)

    violation: Mapped[Violation] = relationship(
        back_populates="evidence"
    )


class DetectionLog(Base):
    __tablename__ = "detection_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int | None] = mapped_column(
        ForeignKey("cameras.id"), index=True
    )
    detected_object: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    ocr_text: Mapped[str | None] = mapped_column(String(30))
    confidence: Mapped[float | None] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False, index=True
    )
    frame_path: Mapped[str | None] = mapped_column(String(500))
    extra_data: Mapped[str | None] = mapped_column(Text)

    camera: Mapped[Camera | None] = relationship(
        back_populates="detection_logs"
    )


Index(
    "ix_violations_datetime_status",
    Violation.violation_datetime,
    Violation.verification_status,
)
