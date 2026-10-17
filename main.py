
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db, SessionLocal
from models import Vehicle, Violation, ViolationType

app = FastAPI(title="Smart Traffic Violation Detection API")

# Allow requests from the React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def home():
    return {"message": "Smart Traffic Violation Detection API is running"}


@app.get("/api/vehicles")
def get_vehicles():
    with SessionLocal() as db:
        vehicles = db.query(Vehicle).all()
        return [
            {
                "id": v.id,
                "vehicle_number": v.vehicle_number,
                "vehicle_type": v.vehicle_type,
                "vehicle_owner": v.vehicle_owner,
                "registered_mobile_number": v.registered_mobile_number,
            }
            for v in vehicles
        ]


@app.get("/api/violations")
def get_violations():
    with SessionLocal() as db:
        violations = db.query(Violation).all()
        return [
            {
                "id": v.id,
                "vehicle_number": (
                    v.vehicle.vehicle_number if v.vehicle else
                    v.detected_vehicle_number
                ),
                "violation_type": v.violation_type.name,
                "violation_datetime": v.violation_datetime.isoformat(),
                "confidence": v.confidence,
                "verification_status": v.verification_status,
            }
            for v in violations
        ]


@app.get("/api/violation-types")
def get_violation_types():
    with SessionLocal() as db:
        types = db.query(ViolationType).filter(
            ViolationType.is_active.is_(True)
        ).all()
        return [
            {"id": t.id, "code": t.code, "name": t.name}
            for t in types
        ]
