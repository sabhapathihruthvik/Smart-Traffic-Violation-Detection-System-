
from contextlib import asynccontextmanager
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from database import SessionLocal, init_db
from inference import (
    convert_video_for_browser,
    new_analysis_id,
    process_media,
    resolve_lane_config,
    resolve_operations,
    resolve_red_light_config,
    validate_operation_media,
)
import inference_config as inference_settings
from models import Vehicle, Violation, ViolationType


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Smart Traffic Violation Detection API",
    lifespan=lifespan,
)
_VIDEO_CONVERSION_LOCK = Lock()
_ANALYSIS_JOBS_LOCK = Lock()
_SAMPLE_METADATA_LOCK = Lock()
_ANALYSIS_JOBS: dict[str, dict] = {}
_SAMPLE_METADATA_PATH = inference_settings.SAMPLE_DIR / "metadata.jsonl"
_CHALLAN_DIR = Path(__file__).resolve().parent / "challans"


class WrongDirectionConfiguration(BaseModel):
    source_key: str
    lane_direction: str
    lane_polygon: list[list[float]]


class RedLightConfiguration(BaseModel):
    source_key: str
    light_roi: list[float]
    stop_line: list[list[float]]
    crossing_direction: str


class ChallanRequest(BaseModel):
    analysis_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    vehicle_id: str = Field(pattern=r"^V\d+$")
    vehicle_number: str = Field(min_length=1, max_length=20)
    violation_code: str = Field(min_length=1, max_length=40)
    fine_amount: int = Field(ge=0, le=1_000_000)


def _read_sample_metadata() -> list[dict]:
    records = []
    try:
        with _SAMPLE_METADATA_PATH.open(encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Sample metadata is invalid at "
                            f"line {line_number}."
                        ),
                    ) from error
                if not isinstance(record, dict):
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Sample metadata must contain JSON objects "
                            f"(line {line_number})."
                        ),
                    )
                records.append(record)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail="Could not read sample metadata."
        ) from error
    return records


def _write_sample_metadata(records: list[dict]) -> None:
    temporary_path = _SAMPLE_METADATA_PATH.with_suffix(".jsonl.tmp")
    try:
        _SAMPLE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        temporary_path.replace(_SAMPLE_METADATA_PATH)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail="Could not save sample metadata."
        ) from error


def _metadata_record_for_source(
    records: list[dict], source_key: str, create: bool = False
) -> dict:
    if not source_key or len(source_key) > 500:
        raise HTTPException(status_code=400, detail="Invalid configuration key.")
    if source_key.startswith("sample:"):
        file_name = source_key.removeprefix("sample:")
        if not file_name or Path(file_name).name != file_name:
            raise HTTPException(status_code=400, detail="Invalid sample key.")
        record = next(
            (item for item in records if item.get("file_name") == file_name),
            None,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="Sample metadata not found.")
        return record

    if source_key.startswith("upload:"):
        record = next(
            (item for item in records if item.get("source_key") == source_key),
            None,
        )
        if record is None and create:
            record = {
                "record_type": "upload_configuration",
                "source_key": source_key,
            }
            records.append(record)
        if record is None:
            raise HTTPException(
                status_code=404, detail="Saved configuration not found."
            )
        return record

    raise HTTPException(status_code=400, detail="Invalid configuration key.")


def _default_wrong_direction_config(records: list[dict]) -> dict:
    defaults = next(
        (item for item in records if item.get("record_type") == "defaults"),
        None,
    )
    config = defaults.get("wrong_direction_config") if defaults else None
    if not isinstance(config, dict):
        raise HTTPException(
            status_code=500,
            detail="Wrong-direction defaults are missing from sample metadata.",
        )
    return _validate_wrong_direction_config(config)


def _default_red_light_config(records: list[dict]) -> dict:
    defaults = next(
        (item for item in records if item.get("record_type") == "defaults"),
        None,
    )
    config = defaults.get("red_light_config") if defaults else None
    if not isinstance(config, dict):
        raise HTTPException(
            status_code=500,
            detail="Red-light defaults are missing from sample metadata.",
        )
    try:
        roi, line, direction = resolve_red_light_config(
            config["light_roi"],
            config["stop_line"],
            config["crossing_direction"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail="Saved red-light defaults are invalid.",
        ) from error
    return {
        "light_roi": roi,
        "stop_line": line,
        "crossing_direction": direction,
    }


def _validate_wrong_direction_config(config: dict) -> dict:
    try:
        direction, polygon = resolve_lane_config(
            config["lane_direction"], config["lane_polygon"]
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail="Saved wrong-direction configuration is invalid.",
        ) from error
    return {"lane_direction": direction, "lane_polygon": polygon}


def _save_red_light_config(
    source_key: str,
    light_roi: list[float] | tuple[float, ...],
    stop_line: list[list[float]] | tuple[tuple[float, float], ...],
    crossing_direction: str,
) -> dict:
    try:
        roi, line, direction = resolve_red_light_config(
            light_roi, stop_line, crossing_direction
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with _SAMPLE_METADATA_LOCK:
        records = _read_sample_metadata()
        record = _metadata_record_for_source(records, source_key, create=True)
        red_light_config = {
            "light_roi": roi,
            "stop_line": line,
            "crossing_direction": direction,
        }
        record["red_light_config"] = red_light_config
        _write_sample_metadata(records)
    return red_light_config


def _save_wrong_direction_config(
    source_key: str,
    lane_direction: str,
    lane_polygon: list[list[float]] | tuple[tuple[float, float], ...],
) -> dict:
    try:
        direction, polygon = resolve_lane_config(lane_direction, lane_polygon)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    with _SAMPLE_METADATA_LOCK:
        records = _read_sample_metadata()
        record = _metadata_record_for_source(records, source_key, create=True)
        wrong_direction_config = {
            "lane_direction": direction,
            "lane_polygon": polygon,
        }
        record["wrong_direction_config"] = wrong_direction_config
        _write_sample_metadata(records)
    return wrong_direction_config


def _update_analysis_job(job_id: str, **updates) -> None:
    with _ANALYSIS_JOBS_LOCK:
        job = _ANALYSIS_JOBS.get(job_id)
        if job is not None:
            job.update(updates)

# Allow requests from the React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_settings.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/results",
    StaticFiles(directory=inference_settings.ANALYSIS_DIR),
    name="analysis-results",
)


@app.get("/")
def home():
    return {"message": "Smart Traffic Violation Detection API is running"}


@app.get("/api/samples")
def get_samples():
    inference_settings.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    records = _read_sample_metadata()
    metadata_by_name = {
        record["file_name"]: record
        for record in records
        if isinstance(record.get("file_name"), str)
    }
    allowed_extensions = (
        inference_settings.IMAGE_EXTENSIONS | inference_settings.VIDEO_EXTENSIONS
    )
    samples = []
    for path in sorted(inference_settings.SAMPLE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed_extensions:
            continue
        metadata = metadata_by_name.get(path.name)
        if metadata is None:
            raise HTTPException(
                status_code=500,
                detail=f"Sample metadata is missing for {path.name}.",
            )
        samples.append({
            "name": path.name,
            "media_type": (
                "video"
                if path.suffix.lower() in inference_settings.VIDEO_EXTENSIONS
                else "image"
            ),
            "url": f"/api/samples/{path.name}",
            "operations_tags": metadata.get("operations_tags", []),
        })
    return samples


@app.get("/api/samples/{filename}")
def get_sample(filename: str):
    sample_path = (inference_settings.SAMPLE_DIR / filename).resolve()
    if (
        sample_path.parent != inference_settings.SAMPLE_DIR.resolve()
        or not sample_path.is_file()
        or sample_path.suffix.lower()
        not in inference_settings.IMAGE_EXTENSIONS | inference_settings.VIDEO_EXTENSIONS
    ):
        raise HTTPException(status_code=404, detail="Sample file not found.")
    return FileResponse(sample_path)


@app.get("/api/wrong-direction/config")
def get_wrong_direction_config(source_key: str | None = None):
    records = _read_sample_metadata()
    default_config = _default_wrong_direction_config(records)
    saved_config = None
    if source_key:
        try:
            record = _metadata_record_for_source(records, source_key)
        except HTTPException as error:
            if error.status_code != 404 or not source_key.startswith("upload:"):
                raise
        else:
            saved_config = record.get("wrong_direction_config")
    config = (
        _validate_wrong_direction_config(saved_config)
        if saved_config is not None
        else default_config
    )
    return {
        "directions": inference_settings.LANE_DIRECTIONS,
        "default_direction": config["lane_direction"],
        "lane_polygon": config["lane_polygon"],
        "saved": saved_config is not None,
    }


@app.put("/api/wrong-direction/config")
def put_wrong_direction_config(configuration: WrongDirectionConfiguration):
    saved_config = _save_wrong_direction_config(
        configuration.source_key,
        configuration.lane_direction,
        configuration.lane_polygon,
    )
    return {"config": saved_config}


@app.get("/api/red-light/config")
def get_red_light_config(source_key: str):
    records = _read_sample_metadata()
    saved_config = None
    try:
        record = _metadata_record_for_source(records, source_key)
    except HTTPException as error:
        if error.status_code != 404 or not source_key.startswith("upload:"):
            raise
    else:
        saved_config = record.get("red_light_config")
    try:
        config = (
            _default_red_light_config(records)
            if saved_config is None
            else {
                "light_roi": saved_config["light_roi"],
                "stop_line": saved_config["stop_line"],
                "crossing_direction": saved_config["crossing_direction"],
            }
        )
        roi, line, direction = resolve_red_light_config(**config)
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail="Saved red-light configuration is invalid.",
        ) from error
    return {
        "config": {
            "light_roi": roi,
            "stop_line": line,
            "crossing_direction": direction,
        }
    }


@app.put("/api/red-light/config")
def put_red_light_config(configuration: RedLightConfiguration):
    saved_config = _save_red_light_config(
        configuration.source_key,
        configuration.light_roi,
        configuration.stop_line,
        configuration.crossing_direction,
    )
    return {"config": saved_config}


def _run_analysis_job(
    job_id: str,
    source_path: Path,
    analysis_dir: Path,
    operations: tuple[str, ...],
    lane_direction: str,
    lane_polygon: tuple[tuple[float, float], ...],
    input_name: str,
    light_roi: tuple[float, float, float, float] | None,
    stop_line: tuple[tuple[float, float], ...] | None,
    crossing_direction: str,
) -> None:
    try:
        _update_analysis_job(
            job_id,
            status="running",
            phase="loading_models",
            progress=0,
            current_frame=0,
            total_frames=None,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        result = process_media(
            source_path,
            analysis_dir,
            operations,
            lane_direction,
            lane_polygon,
            light_roi=light_roi,
            stop_line=stop_line,
            crossing_direction=crossing_direction,
            progress_callback=lambda progress: _update_analysis_job(
                job_id, **progress
            ),
        )
        result["input_name"] = input_name
        result["created_at"] = datetime.now(timezone.utc).isoformat()
        if result["media_type"] == "video":
            result["result_url"] = (
                f"/api/analyses/{analysis_dir.name}/playback"
            )
        (analysis_dir / "metadata.json").write_text(
            json.dumps(result, ensure_ascii=False),
            encoding="utf-8",
        )
        _update_analysis_job(
            job_id,
            status="completed",
            phase="complete",
            progress=100,
            result=result,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as error:
        _update_analysis_job(
            job_id,
            status="failed",
            phase="failed",
            error=str(error),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        raise


@app.get("/api/analysis-jobs/{job_id}")
def get_analysis_job(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    with _ANALYSIS_JOBS_LOCK:
        job = _ANALYSIS_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return dict(job)


@app.get("/api/analyses")
def get_analyses():
    analyses = []
    for analysis_dir in inference_settings.ANALYSIS_DIR.iterdir():
        if not analysis_dir.is_dir() or not re.fullmatch(
            r"[0-9a-f]{32}", analysis_dir.name
        ):
            continue

        outputs = [
            path
            for path in analysis_dir.glob("processed.*")
            if path.is_file()
            and path.suffix.lower()
            in inference_settings.IMAGE_EXTENSIONS
            | inference_settings.VIDEO_EXTENSIONS
        ]
        if not outputs:
            continue
        output_path = outputs[0]
        is_video = output_path.suffix.lower() in inference_settings.VIDEO_EXTENSIONS

        metadata_path = analysis_dir / "metadata.json"
        metadata = {}
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not read analysis metadata: {analysis_dir.name}.",
                ) from error
            if not isinstance(metadata, dict):
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid analysis metadata: {analysis_dir.name}.",
                )

        source_path = next(
            (path for path in analysis_dir.glob("source.*") if path.is_file()),
            None,
        )
        result_url = (
            f"/api/analyses/{analysis_dir.name}/playback"
            if is_video
            else f"/results/{analysis_dir.name}/{output_path.name}"
        )
        analyses.append(
            {
                "analysis_id": analysis_dir.name,
                "media_type": "video" if is_video else "image",
                "input_name": metadata.get(
                    "input_name", source_path.name if source_path else "Unknown file"
                ),
                "result_url": result_url,
                "operations": metadata.get("operations", []),
                "ocr_results": metadata.get("ocr_results", []),
                "wrong_direction_events": metadata.get(
                    "wrong_direction_events", []
                ),
                "red_light_events": metadata.get("red_light_events", []),
                "red_light_signal_state": metadata.get(
                    "red_light_signal_state"
                ),
                "red_light_configuration": metadata.get(
                    "red_light_configuration"
                ),
                "created_at": metadata.get(
                    "created_at",
                    datetime.fromtimestamp(
                        output_path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                ),
            }
        )

    return sorted(analyses, key=lambda item: item["created_at"], reverse=True)


@app.get("/api/analyses/{analysis_id}/playback")
def get_analysis_playback(analysis_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    analysis_dir = (
        inference_settings.ANALYSIS_DIR / analysis_id
    ).resolve()
    if (
        analysis_dir.parent != inference_settings.ANALYSIS_DIR.resolve()
        or not analysis_dir.is_dir()
    ):
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    webm_path = analysis_dir / "processed.webm"
    if webm_path.is_file():
        return FileResponse(webm_path, media_type="video/webm")

    source_path = next(
        (
            path
            for path in analysis_dir.glob("processed.*")
            if path.is_file()
            and path.suffix.lower() in inference_settings.VIDEO_EXTENSIONS
        ),
        None,
    )
    if source_path is None:
        raise HTTPException(status_code=404, detail="Processed video not found.")

    browser_video_path = analysis_dir / "processed-browser.webm"
    with _VIDEO_CONVERSION_LOCK:
        if not browser_video_path.is_file():
            try:
                convert_video_for_browser(source_path, browser_video_path)
            except RuntimeError as error:
                raise HTTPException(status_code=500, detail=str(error)) from error
    return FileResponse(browser_video_path, media_type="video/webm")


@app.post("/api/analyze")
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    sample: str | None = Form(default=None),
    operations: list[str] = Form(default=["vehicle"]),
    configuration_source_key: str | None = Form(default=None),
    red_light_source_key: str | None = Form(default=None),
):
    if (file is None) == (sample is None):
        raise HTTPException(
            status_code=400,
            detail="Choose exactly one uploaded file or sample file.",
        )

    try:
        selected_operations = resolve_operations(operations)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    filename = ""
    sample_metadata = None
    if sample is not None:
        sample_path = (inference_settings.SAMPLE_DIR / sample).resolve()
        if (
            sample_path.parent != inference_settings.SAMPLE_DIR.resolve()
            or not sample_path.is_file()
            or sample_path.suffix.lower()
            not in inference_settings.IMAGE_EXTENSIONS | inference_settings.VIDEO_EXTENSIONS
        ):
            raise HTTPException(status_code=404, detail="Sample file not found.")
        source_path = sample_path
        filename = sample_path.name
        source_suffix = sample_path.suffix.lower()
        sample_metadata = next(
            (
                record
                for record in _read_sample_metadata()
                if record.get("file_name") == filename
            ),
            None,
        )
        if sample_metadata is None:
            raise HTTPException(
                status_code=500,
                detail=f"Sample metadata is missing for {filename}.",
            )
    else:
        filename = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        source_suffix = suffix
        supported_extensions = (
            inference_settings.IMAGE_EXTENSIONS
            | inference_settings.VIDEO_EXTENSIONS
        )
        if suffix not in supported_extensions:
            raise HTTPException(
                status_code=415,
                detail="Upload an image or video in a supported format.",
            )

    try:
        validate_operation_media(source_suffix, selected_operations)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    selected_lane_direction = inference_settings.DEFAULT_LANE_DIRECTION
    selected_lane_polygon = inference_settings.DEFAULT_LANE_POLYGON
    if "wrong_direction" in selected_operations:
        expected_source_key = f"sample:{filename}" if sample is not None else None
        if (
            sample is not None
            and configuration_source_key
            and configuration_source_key != expected_source_key
        ):
            raise HTTPException(
                status_code=400,
                detail="Wrong-direction settings must match the selected source.",
            )
        source_key = expected_source_key or configuration_source_key
        metadata_records = _read_sample_metadata()
        record = None
        if source_key:
            try:
                record = _metadata_record_for_source(metadata_records, source_key)
            except HTTPException as error:
                if error.status_code != 404 or not source_key.startswith("upload:"):
                    raise
        lane_config = (
            record.get("wrong_direction_config") if record is not None else None
        ) or _default_wrong_direction_config(metadata_records)
        try:
            selected_lane_direction, selected_lane_polygon = resolve_lane_config(
                lane_config["lane_direction"],
                lane_config["lane_polygon"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    light_roi = None
    stop_line = None
    selected_crossing_direction = (
        inference_settings.DEFAULT_RED_LIGHT_CROSSING_DIRECTION
    )
    if "red_light" in selected_operations:
        expected_source_key = f"sample:{filename}" if sample is not None else None
        expected_source_key = expected_source_key or configuration_source_key
        if not red_light_source_key:
            red_light_source_key = expected_source_key
        if (
            (red_light_source_key and expected_source_key
             and red_light_source_key != expected_source_key)
            or (configuration_source_key and expected_source_key
                and configuration_source_key != expected_source_key)
        ):
            raise HTTPException(
                status_code=400,
                detail="Red-light settings must match the selected source.",
            )
        metadata_records = _read_sample_metadata()
        record = None
        if red_light_source_key:
            try:
                record = _metadata_record_for_source(
                    metadata_records, red_light_source_key
                )
            except HTTPException as error:
                if error.status_code != 404 or not red_light_source_key.startswith(
                    "upload:"
                ):
                    raise
        red_light_config = (
            record.get("red_light_config") if record is not None else None
        ) or _default_red_light_config(metadata_records)
        try:
            light_roi, stop_line, selected_crossing_direction = (
                resolve_red_light_config(
                    red_light_config["light_roi"],
                    red_light_config["stop_line"],
                    red_light_config["crossing_direction"],
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    analysis_id = new_analysis_id()
    analysis_dir = inference_settings.ANALYSIS_DIR / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=False)

    if file is not None:
        source_path = analysis_dir / filename
        size = 0
        too_large = False
        with source_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > inference_settings.MAX_UPLOAD_BYTES:
                    too_large = True
                    break
                destination.write(chunk)
        await file.close()
        if too_large:
            source_path.unlink(missing_ok=True)
            analysis_dir.rmdir()
            raise HTTPException(
                status_code=413,
                detail="File is larger than the 500 MB upload limit.",
            )

    job_id = new_analysis_id()
    with _ANALYSIS_JOBS_LOCK:
        _ANALYSIS_JOBS[job_id] = {
            "job_id": job_id,
            "analysis_id": analysis_id,
            "status": "queued",
            "phase": "queued",
            "progress": 0,
            "current_frame": 0,
            "total_frames": None,
        }
    background_tasks.add_task(
        _run_analysis_job,
        job_id,
        source_path,
        analysis_dir,
        selected_operations,
        selected_lane_direction,
        selected_lane_polygon,
        filename,
        light_roi,
        stop_line,
        selected_crossing_direction,
    )
    return {"job_id": job_id, "analysis_id": analysis_id, "status": "queued"}


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


@app.get("/api/vehicles/lookup")
def lookup_vehicle(vehicle_number: str):
    normalized_number = vehicle_number.strip().upper()
    if not normalized_number or len(normalized_number) > 20:
        raise HTTPException(status_code=400, detail="Invalid vehicle number.")
    with SessionLocal() as db:
        vehicle = db.connection().exec_driver_sql(
            "SELECT id, vehicle_number, vehicle_type, vehicle_owner, "
            "registered_mobile_number FROM vehicles "
            "WHERE vehicle_number = ? COLLATE NOCASE LIMIT 1",
            (normalized_number,),
        ).mappings().one_or_none()
        if vehicle is None:
            return {"found": False, "vehicle_number": normalized_number}
        return {
            "found": True,
            **dict(vehicle),
        }


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
            {
                "id": t.id,
                "code": t.code,
                "name": t.name,
                "fine_amount": t.fine_amount,
            }
            for t in types
        ]


@app.post("/api/challans")
def create_challan(request: ChallanRequest):
    normalized_number = request.vehicle_number.strip().upper()
    if not normalized_number:
        raise HTTPException(status_code=400, detail="Vehicle number is required.")
    with SessionLocal() as db:
        connection = db.connection()
        vehicle = connection.exec_driver_sql(
            "SELECT vehicle_number, vehicle_type, vehicle_owner, "
            "registered_mobile_number FROM vehicles "
            "WHERE vehicle_number = ? COLLATE NOCASE LIMIT 1",
            (normalized_number,),
        ).mappings().one_or_none()
        if vehicle is None:
            raise HTTPException(
                status_code=404,
                detail="No registered vehicle matches this number.",
            )
        violation_type = connection.exec_driver_sql(
            "SELECT name FROM violation_types "
            "WHERE code = ? AND is_active = 1 LIMIT 1",
            (request.violation_code,),
        ).mappings().one_or_none()
        if violation_type is None:
            raise HTTPException(status_code=404, detail="Violation type not found.")
        vehicle_info = {
            "vehicle_number": vehicle["vehicle_number"],
            "vehicle_type": vehicle["vehicle_type"],
            "vehicle_owner": vehicle["vehicle_owner"] or "Not available",
            "registered_mobile_number": (
                vehicle["registered_mobile_number"] or "Not available"
            ),
        }
        violation_name = violation_type["name"]

    analysis_dir = (
        inference_settings.ANALYSIS_DIR / request.analysis_id
    ).resolve()
    if (
        analysis_dir.parent != inference_settings.ANALYSIS_DIR.resolve()
        or not analysis_dir.is_dir()
    ):
        raise HTTPException(status_code=404, detail="Analysis result not found.")
    metadata_path = analysis_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404, detail="Analysis metadata not found."
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500, detail="Could not read analysis metadata."
        ) from error
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=500, detail="Analysis metadata is invalid.")

    ocr_results = metadata.get("ocr_results", [])
    if not isinstance(ocr_results, list):
        raise HTTPException(
            status_code=500, detail="Analysis OCR metadata is invalid."
        )
    ocr_entry = next(
        (
            entry
            for entry in ocr_results
            if isinstance(entry, dict)
            and entry.get("vehicle_id") == request.vehicle_id
            and entry.get("violation_codes")
        ),
        None,
    )
    if ocr_entry is None:
        raise HTTPException(
            status_code=400,
            detail="This analysis has no OCR-confirmed violation for that vehicle.",
        )
    evidence_path = analysis_dir / "evidence" / f"{request.vehicle_id}.jpg"
    if not evidence_path.is_file():
        raise HTTPException(
            status_code=404, detail="Violation evidence image not found."
        )

    challan_id = uuid4().hex
    evidence_url = (
        f"/results/{request.analysis_id}/evidence/{request.vehicle_id}.jpg"
    )
    issued_at = datetime.now(timezone.utc)
    fields = {
        "challan_id": challan_id,
        "created_at": issued_at.strftime("%d %B %Y · %I:%M %p UTC"),
        "issued_at": issued_at.isoformat(),
        **vehicle_info,
        "violation_type": violation_name,
        "fine_amount": request.fine_amount,
    }
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traffic challan · {html.escape(challan_id)}</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;padding:40px 18px;background:#edf2f5;color:#182538;font:15px/1.5 "Segoe UI",Arial,sans-serif}}
.sheet{{max-width:850px;margin:0 auto;background:#fff;border:1px solid #d9e1e8;border-radius:20px;overflow:hidden;box-shadow:0 24px 70px #192b3b20}}
.masthead{{position:relative;overflow:hidden;padding:32px 38px;color:#fff;background:linear-gradient(120deg,#102235,#183b54 68%,#14665e)}}
.masthead:after{{position:absolute;right:-46px;top:-112px;width:300px;height:300px;border:1px solid #ffffff24;border-radius:50%;content:""}}
.masthead:before{{position:absolute;right:50px;top:-74px;width:220px;height:220px;border:1px solid #ffffff20;border-radius:50%;content:""}}
.brand-row,.title-row,.document-meta,.info-grid,.detail-grid{{display:flex;justify-content:space-between;gap:20px}}
.brand{{display:flex;align-items:center;gap:12px;font-size:14px;font-weight:700;letter-spacing:.13em;text-transform:uppercase}}
.brand-mark{{display:grid;width:38px;height:38px;place-items:center;border:1px solid #ffffff55;border-radius:12px;color:#a8f0d7;font-weight:800}}
.status{{position:relative;z-index:1;align-self:center;padding:7px 12px;border:1px solid #a8f0d788;border-radius:999px;color:#c5f6df;font-size:10px;font-weight:700;letter-spacing:.14em}}
.title-row{{align-items:flex-end;margin-top:40px}}
.eyebrow,.label{{margin:0;color:#7a8a99;font-size:10px;font-weight:700;letter-spacing:.16em;text-transform:uppercase}}
.masthead .eyebrow{{color:#a7c7d2}}
h1{{margin:5px 0 0;font-size:32px;line-height:1.1;letter-spacing:-.04em}}
.fine-box{{position:relative;z-index:1;min-width:190px;padding:14px 18px;border:1px solid #ffffff38;border-radius:14px;background:#ffffff12;text-align:right}}
.fine-box span{{display:block;color:#b8d2dc;font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}}
.fine-box strong{{display:block;margin-top:2px;color:#c5f6df;font-size:27px;line-height:1.2}}
.content{{padding:30px 38px 34px}}
.document-meta{{align-items:center;padding-bottom:22px;border-bottom:1px solid #e8edf1}}
.document-meta strong{{display:block;margin-top:4px;font:600 12px "Consolas",monospace;letter-spacing:.04em}}
.section-title{{display:flex;align-items:center;gap:10px;margin:26px 0 13px;color:#26394c;font-size:13px;font-weight:700}}
.section-title:before{{width:4px;height:18px;border-radius:3px;background:#14a77a;content:""}}
.info-grid{{gap:12px}}
.info-card{{flex:1;min-width:0;padding:15px 16px;border:1px solid #e5ebef;border-radius:13px;background:#f8fafb}}
.info-card span,.detail-item span{{display:block;color:#748493;font-size:9px;font-weight:700;letter-spacing:.12em;text-transform:uppercase}}
.info-card strong{{display:block;margin-top:7px;overflow-wrap:anywhere;color:#192b3d;font-size:14px}}
.vehicle-number{{font:700 19px "Consolas",monospace!important;letter-spacing:.06em}}
.violation-card{{display:grid;grid-template-columns:1fr auto;align-items:center;gap:18px;padding:18px 20px;border:1px solid #dcebe5;border-radius:14px;background:linear-gradient(105deg,#f2faf6,#f5f8fa)}}
.violation-card strong{{display:block;margin-top:6px;color:#183849;font-size:17px}}
.fine-inline{{color:#12684f;font-size:20px;font-weight:800;white-space:nowrap}}
.evidence-frame{{margin:0;overflow:hidden;border:1px solid #e2e8ed;border-radius:14px;background:#f5f7f9}}
.evidence-frame img{{display:block;width:100%;max-height:400px;object-fit:contain;background:#101820}}
.evidence-frame figcaption{{padding:10px 14px;color:#71808d;font-size:10px}}
.footer{{display:flex;justify-content:space-between;gap:16px;margin-top:28px;padding-top:16px;border-top:1px solid #e8edf1;color:#83909b;font-size:10px}}
.footer strong{{color:#426071}}
@media(max-width:620px){{body{{padding:12px}}.sheet{{border-radius:14px}}.masthead,.content{{padding:24px 20px}}.title-row{{align-items:flex-start;flex-direction:column;margin-top:32px}}.fine-box{{width:100%;text-align:left}}.info-grid{{flex-direction:column}}.document-meta{{align-items:flex-start;flex-direction:column}}.footer{{flex-direction:column}}}}
@media print{{@page{{size:A4;margin:12mm}}body{{padding:0;background:#fff;-webkit-print-color-adjust:exact;print-color-adjust:exact}}.sheet{{max-width:none;border:0;border-radius:0;box-shadow:none}}.masthead{{border-radius:0}}.content{{padding:24px 30px}}.evidence-frame img{{max-height:310px}}}}
</style></head>
<body><main class="sheet">
<header class="masthead">
<div class="brand-row"><div class="brand"><span class="brand-mark">TV</span><span>Traffic Vision</span></div><span class="status">● &nbsp; ISSUED</span></div>
<div class="title-row"><div><p class="eyebrow">TRAFFIC ENFORCEMENT</p><h1>Violation challan</h1></div>
<div class="fine-box"><span>Payable fine</span><strong>₹{fields["fine_amount"]:,}</strong></div></div>
</header>
<div class="content">
<div class="document-meta"><div><span class="label">Challan reference</span><strong>{html.escape(fields["challan_id"].upper())}</strong></div>
<div><span class="label">Date issued</span><strong>{html.escape(fields["created_at"])}</strong></div></div>
<h2 class="section-title">Registered vehicle & owner</h2>
<div class="info-grid">
<div class="info-card"><span>Vehicle number</span><strong class="vehicle-number">{html.escape(fields["vehicle_number"])}</strong></div>
<div class="info-card"><span>Vehicle class</span><strong>{html.escape(fields["vehicle_type"])}</strong></div>
<div class="info-card"><span>Registered owner</span><strong>{html.escape(fields["vehicle_owner"])}</strong></div>
<div class="info-card"><span>Contact number</span><strong>{html.escape(fields["registered_mobile_number"])}</strong></div>
</div>
<h2 class="section-title">Violation details</h2>
<div class="violation-card"><div><span class="label">Offence recorded</span><strong>{html.escape(fields["violation_type"])}</strong></div><span class="fine-inline">₹{fields["fine_amount"]:,}</span></div>
<h2 class="section-title">Violation evidence</h2>
<figure class="evidence-frame"><img src="{html.escape(evidence_url, quote=True)}" alt="Captured frame of the traffic violation"><figcaption>Captured frame associated with this vehicle and offence.</figcaption></figure>
<footer class="footer"><span><strong>Traffic Vision</strong> · Smart Traffic Analysis</span><span>Issued {html.escape(fields["created_at"])}</span></footer>
</div></main></body></html>"""

    try:
        _CHALLAN_DIR.mkdir(parents=True, exist_ok=True)
        document_path = _CHALLAN_DIR / f"{challan_id}.html"
        temporary_path = document_path.with_suffix(".html.tmp")
        temporary_path.write_text(document, encoding="utf-8")
        temporary_path.replace(document_path)
    except OSError as error:
        raise HTTPException(
            status_code=500, detail="Could not save the challan."
        ) from error

    return {
        "challan_id": challan_id,
        "status": "sent",
        "message": "Challan sent successfully.",
        "file_url": f"/api/challans/{challan_id}/document",
        "evidence_url": evidence_url,
    }


@app.get("/api/challans/{challan_id}/document")
def get_challan_document(challan_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", challan_id):
        raise HTTPException(status_code=404, detail="Challan not found.")
    document_path = _CHALLAN_DIR / f"{challan_id}.html"
    if not document_path.is_file():
        raise HTTPException(status_code=404, detail="Challan not found.")
    return FileResponse(document_path, media_type="text/html")
