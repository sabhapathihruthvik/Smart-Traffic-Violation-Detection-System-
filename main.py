
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from database import SessionLocal
from inference import (
    convert_video_for_browser,
    new_analysis_id,
    process_media,
    resolve_lane_config,
    resolve_operations,
)
import inference_config as inference_settings
from models import Vehicle, Violation, ViolationType

app = FastAPI(title="Smart Traffic Violation Detection API")
_VIDEO_CONVERSION_LOCK = Lock()

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
    allowed_extensions = (
        inference_settings.IMAGE_EXTENSIONS | inference_settings.VIDEO_EXTENSIONS
    )
    samples = [
        {
            "name": path.name,
            "media_type": (
                "video"
                if path.suffix.lower() in inference_settings.VIDEO_EXTENSIONS
                else "image"
            ),
            "url": f"/api/samples/{path.name}",
        }
        for path in sorted(inference_settings.SAMPLE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]
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
def get_wrong_direction_config():
    return {
        "directions": inference_settings.LANE_DIRECTIONS,
        "default_direction": inference_settings.DEFAULT_LANE_DIRECTION,
        "lane_polygon": inference_settings.DEFAULT_LANE_POLYGON,
    }


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
    file: UploadFile | None = File(default=None),
    sample: str | None = Form(default=None),
    operations: list[str] = Form(default=["vehicle"]),
    lane_direction: str = Form(default=inference_settings.DEFAULT_LANE_DIRECTION),
    lane_polygon: str | None = Form(default=None),
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

    selected_lane_direction = inference_settings.DEFAULT_LANE_DIRECTION
    selected_lane_polygon = inference_settings.DEFAULT_LANE_POLYGON
    if "wrong_direction" in selected_operations:
        try:
            submitted_polygon = (
                json.loads(lane_polygon)
                if lane_polygon is not None
                else inference_settings.DEFAULT_LANE_POLYGON
            )
            selected_lane_direction, selected_lane_polygon = resolve_lane_config(
                lane_direction, submitted_polygon
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

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
    else:
        filename = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1]
        suffix = Path(filename).suffix.lower()
        supported_extensions = (
            inference_settings.IMAGE_EXTENSIONS
            | inference_settings.VIDEO_EXTENSIONS
        )
        if suffix not in supported_extensions:
            raise HTTPException(
                status_code=415,
                detail="Upload an image or video in a supported format.",
            )

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

    try:
        result = await run_in_threadpool(
            process_media,
            source_path,
            analysis_dir,
            selected_operations,
            selected_lane_direction,
            selected_lane_polygon,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    result["input_name"] = source_path.name
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    if result["media_type"] == "video":
        result["result_url"] = f"/api/analyses/{analysis_id}/playback"
    metadata_path = analysis_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


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
