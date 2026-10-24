# Smart-Traffic-Violation-Detection-System-

## Combined image/video analysis

The home page lets you upload an image or video, select media from
`data/sample_data`, and choose vehicle detection, helmet detection, number-plate
detection, OCR, and optional wrong-direction detection. Helmet/plate detection
and wrong-direction detection automatically require vehicle tracking; OCR
automatically enables plate and vehicle detection.
Processed media is saved under a unique `analysis_results/<analysis-id>/`
folder. Processed videos use browser-playable WebM/VP8 output, while existing
MP4 results are converted to WebM the first time they are played in the UI.
The UI keeps a collapsible history of completed analyses. OCR results are
returned with their vehicle tracking ID; analysis results are not written to
the database.

Wrong-direction detection uses persistent vehicle tracks and a manually
configured lane polygon, not a separate detector. Choose the permitted lane
direction and edit the polygon's four normalized frame points in the UI.
Thresholds, trajectory history, default lane geometry, and debug overlays are
configurable in `inference_config.py`.

Place the trained weights in the root `models/` folder using these filenames:

- `yolo26m.pt` — vehicle detection and ByteTrack
- `helmet_detector2.pt` — helmet detection within bike/motorcycle regions
- `yolo26m_face_lp.pt` — license-plate detection

RapidOCR performs plate text recognition. Detection thresholds, image sizes,
tracked vehicle classes, and model filenames are centralized in
`inference_config.py`. Crop detectors are batched per frame, video decoding is
prefetched on a reader thread, and models are loaded only for selected
operations.

Run the API from the repository root:

```powershell
uv run uvicorn main:app --reload
```

In another terminal, run the frontend:

```powershell
Set-Location .\frontend
npm run dev
```