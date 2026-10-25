# Smart-Traffic-Violation-Detection-System-

## Combined image/video analysis

The home page lets you upload an image or video, select media from
`data/sample_data`, and choose vehicle detection, helmet detection, number-plate
detection, OCR, wrong-direction detection, and red-light detection. Helmet,
plate, OCR, and violation detection automatically require vehicle tracking; OCR
requires vehicle tracking but can be selected independently of number-plate
detection. When any violation operation is selected, plate detection and OCR
are run only on vehicles confirmed as violating. OCR uses the detected plate
crop when number-plate detection is selected, or the full vehicle crop otherwise.
For a continuing violation, selected plate detection and OCR retry immediately
and then every five video frames while the vehicle remains flagged.
Plate/OCR-only analysis is supported on images; videos require at least one
violation operation when plate detection or OCR is selected.
Processed media is saved under a unique `analysis_results/<analysis-id>/`
folder. Processed videos use browser-playable WebM/VP8 output, while existing
MP4 results are converted to WebM the first time they are played in the UI.
The UI keeps a collapsible history of completed analyses. OCR results are
returned with their vehicle tracking ID; analysis results are not written to
the database.

Wrong-direction detection uses persistent vehicle tracks and a manually
configured lane polygon, not a separate detector. Choose the permitted lane
direction and edit the polygon's normalized frame points in the UI, then save
the settings for that source.

Red-light detection is an optional operation that reuses vehicle tracking.
Select it for a video, draw the traffic-light ROI and stop line over the source
preview, choose the crossing direction, and save the setup. The normalized
coordinates and wrong-direction settings are read from and saved to
`data/sample_data/metadata.jsonl`, which also provides defaults for sources
without saved settings. Operation tags in that file are suggestions only;
every operation can be selected for any supported image or video. The result
overlays the detected signal state and marks confirmed red-light crossings.
During analysis, the UI displays the current processing phase, frame-based
progress, and an estimated remaining time when the video reports a frame count.

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
Set-Location .\frontend or `cd frontend`
npm run dev
```