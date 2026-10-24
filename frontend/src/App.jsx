
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const ACCEPTED_MEDIA =
  ".jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.mp4,.avi,.mov,.mkv,.webm,.m4v";
const fetchAnalyses = async () => {
  const response = await fetch(`${API_BASE}/api/analyses`);
  if (!response.ok) throw new Error("Could not load previous analyses.");
  return response.json();
};
const DEFAULT_LANE_DIRECTIONS = [
  "TOP_TO_BOTTOM",
  "BOTTOM_TO_TOP",
  "LEFT_TO_RIGHT",
  "RIGHT_TO_LEFT",
];
const DEFAULT_LANE_POLYGON = [
  [43, 8],
  [57, 8],
  [90, 98],
  [10, 98],
];
const VIDEO_EXTENSIONS = new Set([
  ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
]);
const RED_LIGHT_CROSSING_DIRECTIONS = [
  "NEGATIVE_TO_POSITIVE",
  "POSITIVE_TO_NEGATIVE",
];
const formatDirection = (direction) => direction.replace("_TO_", " → ");
const OPERATIONS = [
  {
    id: "vehicle",
    title: "Vehicle detection",
    description: "Find and track vehicles with unique IDs.",
    dependency: "Base operation",
  },
  {
    id: "helmet",
    title: "Helmet detection",
    description: "Check helmet use inside bike/motorcycle boxes.",
    dependency: "Requires vehicle detection",
  },
  {
    id: "plate",
    title: "Number plate detection",
    description: "Locate plates belonging to tracked vehicles.",
    dependency: "Requires vehicle detection",
  },
  {
    id: "ocr",
    title: "OCR",
    description: "Read text from detected number plates.",
    dependency: "Requires plate + vehicle detection",
  },
  {
    id: "wrong_direction",
    title: "Wrong direction detection",
    description: "Flag tracked vehicles moving against the permitted lane direction.",
    dependency: "Requires vehicle detection",
  },
  {
    id: "red_light",
    title: "Red-light violation detection",
    description: "Detect tracked vehicles crossing a configured stop line on red.",
    dependency: "Requires vehicle tracking · Video only",
  },
];

function App() {
  const [samples, setSamples] = useState([]);
  const [analyses, setAnalyses] = useState([]);
  const [laneDirections, setLaneDirections] = useState(DEFAULT_LANE_DIRECTIONS);
  const [samplesError, setSamplesError] = useState("");
  const [analysesError, setAnalysesError] = useState("");
  const [laneConfigError, setLaneConfigError] = useState("");
  const [selectedSample, setSelectedSample] = useState("");
  const [upload, setUpload] = useState(null);
  const [selectedOperations, setSelectedOperations] = useState(
    () => new Set(["vehicle"]),
  );
  const [laneDirection, setLaneDirection] = useState("BOTTOM_TO_TOP");
  const [lanePolygon, setLanePolygon] = useState(DEFAULT_LANE_POLYGON);
  const [laneConfigSaved, setLaneConfigSaved] = useState(false);
  const [laneConfigLoading, setLaneConfigLoading] = useState(false);
  const [redLightRoi, setRedLightRoi] = useState(null);
  const [redLightStopLine, setRedLightStopLine] = useState(null);
  const [redLightCrossingDirection, setRedLightCrossingDirection] = useState(
    RED_LIGHT_CROSSING_DIRECTIONS[0],
  );
  const [redLightConfigSaved, setRedLightConfigSaved] = useState(false);
  const [redLightConfigLoading, setRedLightConfigLoading] = useState(false);
  const [redLightConfigError, setRedLightConfigError] = useState("");
  const [redLightDrawingMode, setRedLightDrawingMode] = useState("light");
  const [redLightDraft, setRedLightDraft] = useState(null);
  const [redLightPreviewRatio, setRedLightPreviewRatio] = useState(16 / 9);
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState("");
  const uploadPreviewUrlRef = useRef("");
  const [analysisJob, setAnalysisJob] = useState(null);
  const [uploading, setUploading] = useState(false);
  const uploadStartedAt = useRef(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  const selectedSampleInfo = samples.find(
    (sample) => sample.name === selectedSample,
  );
  const redLightSourceKey = upload
    ? `upload:${upload.name}:${upload.size}:${upload.lastModified}`
    : selectedSample
      ? `sample:${selectedSample}`
      : "";
  const isSelectedVideo = upload
    ? VIDEO_EXTENSIONS.has(
        `.${upload.name.split(".").pop()?.toLowerCase() || ""}`,
      )
    : selectedSampleInfo?.media_type === "video";
  const sourcePreviewUrl = upload
    ? uploadPreviewUrl
    : selectedSampleInfo
      ? new URL(selectedSampleInfo.url, API_BASE).toString()
      : "";

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/samples`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Could not load sample files.");
        return response.json();
      })
      .then((items) => {
        if (active) setSamples(items);
      })
      .catch((fetchError) => {
        if (active) setSamplesError(fetchError.message);
      });

    fetchAnalyses()
      .then((items) => {
        if (active) setAnalyses(items);
      })
      .catch((fetchError) => {
        if (active) setAnalysesError(fetchError.message);
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(
    () => () => {
      if (uploadPreviewUrlRef.current) {
        URL.revokeObjectURL(uploadPreviewUrlRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    if (!selectedOperations.has("wrong_direction") || !redLightSourceKey) return;
    let active = true;
    fetch(
      `${API_BASE}/api/wrong-direction/config?${new URLSearchParams({
        source_key: redLightSourceKey,
      })}`,
    )
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(
            payload.detail || "Could not load wrong-direction settings.",
          );
        }
        return payload;
      })
      .then((config) => {
        if (!active) return;
        setLaneDirections(config.directions);
        setLaneDirection(config.default_direction);
        setLanePolygon(config.lane_polygon.map(([x, y]) => [x * 100, y * 100]));
        setLaneConfigSaved(Boolean(config.saved));
        setLaneConfigError("");
      })
      .catch((fetchError) => {
        if (active) setLaneConfigError(fetchError.message);
      })
      .finally(() => {
        if (active) setLaneConfigLoading(false);
      });
    return () => {
      active = false;
    };
  }, [redLightSourceKey, selectedOperations]);

  useEffect(() => {
    if (!selectedOperations.has("red_light") || !redLightSourceKey) return;
    let active = true;
    fetch(
      `${API_BASE}/api/red-light/config?${new URLSearchParams({
        source_key: redLightSourceKey,
      })}`,
    )
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Could not load saved red-light settings.");
        }
        return payload.config;
      })
      .then((savedConfig) => {
        if (!active) return;
        setRedLightRoi(savedConfig?.light_roi || null);
        setRedLightStopLine(savedConfig?.stop_line || null);
        setRedLightCrossingDirection(
          savedConfig?.crossing_direction || RED_LIGHT_CROSSING_DIRECTIONS[0],
        );
        setRedLightConfigSaved(Boolean(savedConfig));
      })
      .catch((fetchError) => {
        if (!active) return;
        setRedLightRoi(null);
        setRedLightStopLine(null);
        setRedLightConfigError(fetchError.message);
      })
      .finally(() => {
        if (active) setRedLightConfigLoading(false);
      });
    return () => {
      active = false;
    };
  }, [redLightSourceKey, selectedOperations]);

  useEffect(() => {
    if (!isProcessing) return undefined;
    const startedAt = analysisJob?.started_at
      ? new Date(analysisJob.started_at).getTime()
      : uploadStartedAt.current || Date.now();
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.max(0, (Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [analysisJob?.started_at, isProcessing]);

  function replaceUploadPreview(file) {
    if (uploadPreviewUrlRef.current) {
      URL.revokeObjectURL(uploadPreviewUrlRef.current);
    }
    const previewUrl = file ? URL.createObjectURL(file) : "";
    uploadPreviewUrlRef.current = previewUrl;
    setUploadPreviewUrl(previewUrl);
  }

  function chooseSample(name) {
    setSelectedSample(name);
    setUpload(null);
    replaceUploadPreview(null);
    setResult(null);
    setError("");
    setRedLightRoi(null);
    setRedLightStopLine(null);
    setRedLightCrossingDirection(RED_LIGHT_CROSSING_DIRECTIONS[0]);
    setRedLightDraft(null);
    setRedLightConfigError("");
    setRedLightConfigSaved(false);
    setRedLightConfigLoading(selectedOperations.has("red_light"));
    setLaneConfigSaved(false);
    setLaneConfigLoading(selectedOperations.has("wrong_direction"));
    setLaneConfigError("");
  }

  function chooseUpload(event) {
    const nextFile = event.target.files?.[0] || null;
    event.target.value = "";
    setUpload(nextFile);
    replaceUploadPreview(nextFile);
    setSelectedSample("");
    setResult(null);
    setError("");
    setRedLightRoi(null);
    setRedLightStopLine(null);
    setRedLightCrossingDirection(RED_LIGHT_CROSSING_DIRECTIONS[0]);
    setRedLightDraft(null);
    setRedLightConfigError("");
    setRedLightConfigSaved(false);
    setRedLightConfigLoading(
      Boolean(nextFile && selectedOperations.has("red_light")),
    );
    setLaneConfigSaved(false);
    setLaneConfigLoading(
      Boolean(nextFile && selectedOperations.has("wrong_direction")),
    );
    setLaneConfigError("");
  }

  function dropUpload(event) {
    event.preventDefault();
    const nextFile = event.dataTransfer.files?.[0] || null;
    if (!nextFile) return;
    setUpload(nextFile);
    replaceUploadPreview(nextFile);
    setSelectedSample("");
    setResult(null);
    setError("");
    setRedLightRoi(null);
    setRedLightStopLine(null);
    setRedLightCrossingDirection(RED_LIGHT_CROSSING_DIRECTIONS[0]);
    setRedLightDraft(null);
    setRedLightConfigError("");
    setRedLightConfigSaved(false);
    setRedLightConfigLoading(selectedOperations.has("red_light"));
    setLaneConfigSaved(false);
    setLaneConfigLoading(selectedOperations.has("wrong_direction"));
    setLaneConfigError("");
  }

  function toggleOperation(operation, checked) {
    const next = new Set(selectedOperations);
    if (checked) {
      next.add(operation);
      if (
        operation === "helmet" ||
        operation === "plate" ||
        operation === "ocr" ||
        operation === "wrong_direction" ||
        operation === "red_light"
      ) {
        next.add("vehicle");
      }

      if (operation === "red_light") {
        setRedLightConfigSaved(false);
        setRedLightConfigError("");
        setRedLightConfigLoading(Boolean(redLightSourceKey));
      }
      if (operation === "wrong_direction") {
        setLaneConfigSaved(false);
        setLaneConfigLoading(Boolean(redLightSourceKey));
      }
      if (operation === "ocr") next.add("plate");
    } else {
      next.delete(operation);
      if (operation === "vehicle") {
        next.delete("helmet");
        next.delete("plate");
        next.delete("ocr");
        next.delete("wrong_direction");
        next.delete("red_light");
      }
      if (operation === "red_light") {
        setRedLightConfigLoading(false);
      }
      if (operation === "wrong_direction") {
        setLaneConfigLoading(false);
      }
      if (operation === "plate") next.delete("ocr");
    }
    setSelectedOperations(next);
    setResult(null);
    setError("");
  }

  async function saveWrongDirectionConfiguration() {
    if (!redLightSourceKey) {
      setLaneConfigError("Choose a sample or upload a file first.");
      return;
    }
    setLaneConfigLoading(true);
    setLaneConfigError("");
    try {
      const response = await fetch(`${API_BASE}/api/wrong-direction/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_key: redLightSourceKey,
          lane_direction: laneDirection,
          lane_polygon: lanePolygon.map(([x, y]) => [
            Number(x) / 100,
            Number(y) / 100,
          ]),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(
          payload.detail || "Could not save wrong-direction settings.",
        );
      }
      setLaneConfigSaved(true);
    } catch (saveError) {
      setLaneConfigError(saveError.message);
    } finally {
      setLaneConfigLoading(false);
    }
  }

  function getRedLightPoint(event) {
    const bounds = event.currentTarget.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height)),
    ];
  }

  function startRedLightDrawing(event) {
    if (!redLightDrawingMode || !sourcePreviewUrl) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const point = getRedLightPoint(event);
    setRedLightDraft({ start: point, end: point });
  }

  function moveRedLightDrawing(event) {
    if (!redLightDraft) return;
    const end = getRedLightPoint(event);
    setRedLightDraft((draft) => ({
      ...draft,
      end,
    }));
  }

  function finishRedLightDrawing(event) {
    if (!redLightDraft) return;
    const end = getRedLightPoint(event);
    if (redLightDrawingMode === "light") {
      setRedLightRoi([
        Math.min(redLightDraft.start[0], end[0]),
        Math.min(redLightDraft.start[1], end[1]),
        Math.max(redLightDraft.start[0], end[0]),
        Math.max(redLightDraft.start[1], end[1]),
      ]);
    } else {
      setRedLightStopLine([redLightDraft.start, end]);
    }
    setRedLightConfigSaved(false);
    setRedLightDraft(null);
  }

  function resetRedLightConfiguration() {
    setRedLightRoi(null);
    setRedLightStopLine(null);
    setRedLightConfigSaved(false);
    setRedLightConfigError("");
  }

  async function saveRedLightConfiguration() {
    if (
      !redLightSourceKey ||
      !redLightRoi ||
      !redLightStopLine
    ) {
      setRedLightConfigError(
        "Choose a video, draw the light region and stop line, then save.",
      );
      return;
    }
    setRedLightConfigLoading(true);
    setRedLightConfigError("");
    try {
      const response = await fetch(`${API_BASE}/api/red-light/config`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_key: redLightSourceKey,
          light_roi: redLightRoi,
          stop_line: redLightStopLine,
          crossing_direction: redLightCrossingDirection,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Could not save red-light settings.");
      }
      setRedLightConfigSaved(true);
    } catch (saveError) {
      setRedLightConfigError(saveError.message);
    } finally {
      setRedLightConfigLoading(false);
    }
  }

  async function analyze(event) {
    event.preventDefault();
    if (!upload && !selectedSample) {
      setError("Choose a sample or upload an image/video to continue.");
      return;
    }
    const body = new FormData();
    if (upload) body.append("file", upload);
    else body.append("sample", selectedSample);
    selectedOperations.forEach((operation) => body.append("operations", operation));
    if (selectedOperations.has("red_light")) {
      body.append("red_light_source_key", redLightSourceKey);
    }
    if (
      selectedOperations.has("wrong_direction") ||
      selectedOperations.has("red_light")
    ) {
      body.append("configuration_source_key", redLightSourceKey);
    }

    setIsProcessing(true);
    setUploading(Boolean(upload));
    setError("");
    setResult(null);
    setAnalysisJob(null);
    uploadStartedAt.current = Date.now();
    setElapsedSeconds(0);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }
      setAnalysisJob({
        ...payload,
        status: "queued",
        progress: 0,
      });
    } catch (requestError) {
      setError(requestError.message || "Could not connect to the analysis server.");
      setIsProcessing(false);
    } finally {
      setUploading(false);
    }
  }

  useEffect(() => {
    if (!analysisJob?.job_id) return undefined;
    let active = true;
    let timer;
    const pollStatus = async () => {
      try {
        const response = await fetch(
          `${API_BASE}/api/analysis-jobs/${analysisJob.job_id}`,
        );
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || "Could not read analysis progress.");
        }
        if (!active) return;
        setAnalysisJob(payload);
        if (payload.status === "completed") {
          const completedResult = payload.result;
          setResult(completedResult);
          setAnalyses((previous) => [
            completedResult,
            ...previous.filter(
              (item) => item.analysis_id !== completedResult.analysis_id,
            ),
          ]);
          setIsProcessing(false);
          return;
        }
        if (payload.status === "failed") {
          setError(payload.error || "Analysis failed.");
          setIsProcessing(false);
          return;
        }
        timer = window.setTimeout(pollStatus, 1000);
      } catch (pollError) {
        if (!active) return;
        setError(pollError.message);
        setIsProcessing(false);
      }
    };
    timer = window.setTimeout(pollStatus, 350);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [analysisJob?.job_id]);

  const processedFrames = analysisJob?.current_frame || 0;
  const totalFrames = analysisJob?.total_frames;
  const estimatedRemainingSeconds =
    processedFrames > 2 && totalFrames > processedFrames
      ? (elapsedSeconds / processedFrames) * (totalFrames - processedFrames)
      : null;

  function formatDuration(seconds) {
    const rounded = Math.ceil(seconds);
    const minutes = Math.floor(rounded / 60);
    const remainder = rounded % 60;
    return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
  }

  const resultUrl = result
    ? new URL(result.result_url, API_BASE).toString()
    : "";

  function showAnalysis(analysis) {
    setResult(analysis);
    setError("");
  }

  function updateLanePoint(pointIndex, coordinateIndex, value) {
    setLaneConfigSaved(false);
    setLanePolygon((previous) =>
      previous.map((point, index) =>
        index === pointIndex
          ? point.map((coordinate, axis) =>
              axis === coordinateIndex ? value : coordinate,
            )
          : point,
      ),
    );
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="Traffic Vision home">
          <span className="brand-mark" aria-hidden="true">TV</span>
          <span>Traffic<span className="brand-light">Vision</span></span>
        </a>
        <span className="status-pill"><span /> AI ANALYSIS WORKSPACE</span>
      </header>

      <section className="hero" id="home">
        <div className="hero-copy">
          <p className="eyebrow">SMART TRAFFIC ANALYSIS</p>
          <h1>See the road<br /><span>more clearly.</span></h1>
          <p className="hero-description">
            Detect helmet use, track vehicles, and read number plates from a
            single image or video.
          </p>
          <div className="feature-list">
            <span><i>01</i> Helmet detection</span>
            <span><i>02</i> Vehicle tracking</span>
            <span><i>03</i> Plate OCR</span>
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="visual-grid" />
          <div className="road-line road-line-one" />
          <div className="road-line road-line-two" />
          <div className="scan-card scan-card-top">
            <span className="scan-dot" /> VEHICLE TRACKING
            <strong>ACTIVE</strong>
          </div>
          <div className="scan-frame">
            <div className="scan-corner top-left" />
            <div className="scan-corner top-right" />
            <div className="scan-corner bottom-left" />
            <div className="scan-corner bottom-right" />
            <div className="scan-car"><span /></div>
            <div className="scan-label">VEHICLE · V01</div>
            <div className="scan-plate">OCR READY</div>
          </div>
          <div className="scan-card scan-card-bottom">
            <span>PROCESSING PIPELINE</span>
            <div><b /> VEHICLES <b /> HELMETS <b /> PLATES</div>
          </div>
        </div>
      </section>

      <section className="workspace" aria-labelledby="analysis-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">START AN ANALYSIS</p>
            <h2 id="analysis-heading">Choose your footage</h2>
          </div>
          <span className="supported-formats">IMAGE OR VIDEO</span>
        </div>

        <form onSubmit={analyze}>
          <label
            className="upload-area"
            onDragOver={(event) => event.preventDefault()}
            onDrop={dropUpload}
          >
            <input
              type="file"
              accept={ACCEPTED_MEDIA}
              onChange={chooseUpload}
            />
            <span className="upload-icon" aria-hidden="true">↑</span>
            <span className="upload-title">
              {upload ? upload.name : "Drop a file here or browse"}
            </span>
            <span className="upload-hint">Images and videos up to 500 MB</span>
            <span className="browse-button">Browse files</span>
          </label>

          <div className="operation-section">
            <div className="sample-heading">
              <div>
                <h3>Select analysis operations</h3>
                <p>Dependencies are selected automatically.</p>
              </div>
              <span>{selectedOperations.size} SELECTED</span>
            </div>
            <div className="operation-grid">
              {OPERATIONS.map((operation) => {
                const isVehicleRequired =
                  operation.id === "vehicle" &&
                  (selectedOperations.has("helmet") ||
                    selectedOperations.has("plate") ||
                    selectedOperations.has("wrong_direction") ||
                    selectedOperations.has("red_light"));
                const isPlateRequired =
                  operation.id === "plate" && selectedOperations.has("ocr");
                return (
                  <label
                    className={`operation-card ${selectedOperations.has(operation.id) ? "selected" : ""}`}
                    key={operation.id}
                  >
                    <input
                      type="checkbox"
                      checked={selectedOperations.has(operation.id)}
                      disabled={isVehicleRequired || isPlateRequired}
                      onChange={(event) =>
                        toggleOperation(operation.id, event.target.checked)
                      }
                    />
                    <span className="operation-check" aria-hidden="true">✓</span>
                    <span className="operation-copy">
                      <strong>{operation.title}</strong>
                      <small>{operation.description}</small>
                      <em>{operation.dependency}</em>
                    </span>
                  </label>
                );
              })}
            </div>
            {selectedOperations.has("wrong_direction") && (
              <div className="lane-settings">
                <label className="lane-direction-setting">
                  <span>Permitted lane direction</span>
                  <select
                    value={laneDirection}
                    onChange={(event) => {
                      setLaneDirection(event.target.value);
                      setLaneConfigSaved(false);
                    }}
                  >
                    {laneDirections.map((direction) => (
                      <option key={direction} value={direction}>
                        {formatDirection(direction)}
                      </option>
                    ))}
                  </select>
                </label>
                <fieldset className="lane-polygon-setting">
                  <legend>Lane polygon · normalized frame coordinates (%)</legend>
                  <p>
                    Set the polygon points clockwise: top-left, top-right,
                    bottom-right, bottom-left.
                  </p>
                  <div className="lane-point-grid">
                    {lanePolygon.map(([x, y], index) => (
                      <div className="lane-point" key={index}>
                        <strong>Point {index + 1}</strong>
                        <label>
                          X
                          <input
                            type="number"
                            min="0"
                            max="100"
                            step="1"
                            required
                            value={x}
                            onChange={(event) =>
                              updateLanePoint(
                                index,
                                0,
                                event.target.value === ""
                                  ? ""
                                  : Number(event.target.value),
                              )
                            }
                          />
                        </label>
                        <label>
                          Y
                          <input
                            type="number"
                            min="0"
                            max="100"
                            step="1"
                            required
                            value={y}
                            onChange={(event) =>
                              updateLanePoint(
                                index,
                                1,
                                event.target.value === ""
                                  ? ""
                                  : Number(event.target.value),
                              )
                            }
                          />
                        </label>
                      </div>
                    ))}
                  </div>
                  {laneConfigError && (
                    <p className="lane-config-error" role="status">
                      {laneConfigError}
                    </p>
                  )}
                  <div className="lane-config-save-row">
                    <button
                      className="secondary"
                      type="button"
                      onClick={saveWrongDirectionConfiguration}
                      disabled={laneConfigLoading}
                    >
                      {laneConfigLoading
                        ? "Loading settings…"
                        : "Save wrong-direction settings"}
                    </button>
                    <span>
                      {laneConfigSaved
                        ? "Settings saved for this source."
                        : "Changes are not saved yet."}
                    </span>
                  </div>
                </fieldset>
              </div>
            )}
            {selectedOperations.has("red_light") && (
              <div className="red-light-settings">
                <div className="red-light-settings-heading">
                  <div>
                    <h4>Traffic-light and stop-line setup</h4>
                    <p>
                      Draw on the source preview. Coordinates scale to the
                      original frame resolution.
                    </p>
                  </div>
                  <span className={redLightConfigSaved ? "config-saved" : ""}>
                    {redLightConfigLoading
                      ? "LOADING"
                      : redLightConfigSaved
                        ? "SAVED FOR THIS VIDEO"
                        : "NOT SAVED"}
                  </span>
                </div>
                {sourcePreviewUrl ? (
                  <>
                    <div className="red-light-tools">
                      <button
                        className={redLightDrawingMode === "light" ? "active" : ""}
                        type="button"
                        onClick={() => setRedLightDrawingMode("light")}
                      >
                        Draw traffic-light ROI
                      </button>
                      <button
                        className={redLightDrawingMode === "line" ? "active" : ""}
                        type="button"
                        onClick={() => setRedLightDrawingMode("line")}
                      >
                        Draw stop line
                      </button>
                      <button
                        className="secondary"
                        type="button"
                        onClick={resetRedLightConfiguration}
                      >
                        Reset
                      </button>
                    </div>
                    {sourcePreviewUrl ? (
                      <div className="red-light-preview-stage">
                        <div
                          className="red-light-preview-frame"
                          style={{
                            aspectRatio: redLightPreviewRatio,
                            width: `min(100%, ${Math.min(920, 520 * redLightPreviewRatio)}px)`,
                          }}
                        >
                          {isSelectedVideo ? (
                            <video
                              className="red-light-source-preview"
                              src={sourcePreviewUrl}
                              onLoadedMetadata={(event) => {
                                const video = event.currentTarget;
                                if (video.videoWidth && video.videoHeight) {
                                  setRedLightPreviewRatio(
                                    video.videoWidth / video.videoHeight,
                                  );
                                }
                              }}
                              controls
                              muted
                              playsInline
                              preload="metadata"
                            />
                          ) : (
                            <img
                              className="red-light-source-preview"
                              src={sourcePreviewUrl}
                              alt="Selected source preview"
                              onLoad={(event) => {
                                const image = event.currentTarget;
                                if (image.naturalWidth && image.naturalHeight) {
                                  setRedLightPreviewRatio(
                                    image.naturalWidth / image.naturalHeight,
                                  );
                                }
                              }}
                            />
                          )}
                          <svg
                            className="red-light-drawing-overlay"
                            viewBox="0 0 100 100"
                            preserveAspectRatio="none"
                            onPointerDown={startRedLightDrawing}
                            onPointerMove={moveRedLightDrawing}
                            onPointerUp={finishRedLightDrawing}
                            onPointerCancel={() => setRedLightDraft(null)}
                            aria-label="Draw the traffic-light area or stop line on the source"
                          >
                            {redLightRoi && (
                              <rect
                                className="traffic-light-roi"
                                x={redLightRoi[0] * 100}
                                y={redLightRoi[1] * 100}
                                width={(redLightRoi[2] - redLightRoi[0]) * 100}
                                height={(redLightRoi[3] - redLightRoi[1]) * 100}
                              />
                            )}
                            {redLightStopLine && (
                              <line
                                className="configured-stop-line"
                                x1={redLightStopLine[0][0] * 100}
                                y1={redLightStopLine[0][1] * 100}
                                x2={redLightStopLine[1][0] * 100}
                                y2={redLightStopLine[1][1] * 100}
                              />
                            )}
                            {redLightDraft && (
                              redLightDrawingMode === "light" ? (
                                <rect
                                  className="draft-traffic-light-roi"
                                  x={Math.min(
                                    redLightDraft.start[0],
                                    redLightDraft.end[0],
                                  ) * 100}
                                  y={Math.min(
                                    redLightDraft.start[1],
                                    redLightDraft.end[1],
                                  ) * 100}
                                  width={Math.abs(
                                    redLightDraft.end[0] -
                                      redLightDraft.start[0],
                                  ) * 100}
                                  height={Math.abs(
                                    redLightDraft.end[1] -
                                      redLightDraft.start[1],
                                  ) * 100}
                                />
                              ) : (
                                <line
                                  className="draft-stop-line"
                                  x1={redLightDraft.start[0] * 100}
                                  y1={redLightDraft.start[1] * 100}
                                  x2={redLightDraft.end[0] * 100}
                                  y2={redLightDraft.end[1] * 100}
                                />
                              )
                            )}
                          </svg>
                        </div>
                      </div>
                    ) : (
                      <p className="empty-samples">Preparing source preview…</p>
                    )}
                    <p className="red-light-instructions">
                      {redLightDrawingMode === "light"
                        ? "Click and drag around all three signal lamps."
                        : "Drag from one end of the stop line to the other."}
                    </p>
                    <label className="lane-direction-setting">
                      <span>Vehicle crossing direction</span>
                      <select
                        value={redLightCrossingDirection}
                        onChange={(event) => {
                          setRedLightCrossingDirection(event.target.value);
                          setRedLightConfigSaved(false);
                        }}
                      >
                        <option value="NEGATIVE_TO_POSITIVE">
                          Side − to side + (arrow shown on processed video)
                        </option>
                        <option value="POSITIVE_TO_NEGATIVE">
                          Side + to side −
                        </option>
                      </select>
                    </label>
                    {redLightConfigError && (
                      <p className="inline-error" role="alert">
                        {redLightConfigError}
                      </p>
                    )}
                    <div className="red-light-save-row">
                      <button
                        className="analyze-button"
                        type="button"
                        onClick={saveRedLightConfiguration}
                        disabled={
                          redLightConfigLoading ||
                          !redLightRoi ||
                          !redLightStopLine
                        }
                      >
                        Save settings for this source
                      </button>
                      <span>
                        {redLightRoi && redLightStopLine
                          ? "Light ROI and stop line are set."
                          : "Draw both the light ROI and stop line."}
                      </span>
                    </div>
                  </>
                ) : (
                  <p className="empty-samples">Preparing source preview…</p>
                )}
              </div>
            )}
          </div>

          <div className="sample-section">
            <div className="sample-heading">
              <div>
                <h3>Try a sample</h3>
                <p>Quick-start with footage already in the project</p>
              </div>
              <span>{samples.length} FILE{samples.length === 1 ? "" : "S"}</span>
            </div>
            {samplesError ? (
              <p className="inline-error">{samplesError}</p>
            ) : samples.length ? (
              <div className="sample-grid">
                {samples.map((sample) => (
                  <button
                    className={`sample-card ${selectedSample === sample.name ? "selected" : ""}`}
                    key={sample.name}
                    type="button"
                    onClick={() => chooseSample(sample.name)}
                    aria-pressed={selectedSample === sample.name}
                  >
                    <span className="sample-icon" aria-hidden="true">
                      {sample.media_type === "video" ? "▶" : "▧"}
                    </span>
                    <span className="sample-info">
                      <strong title={sample.name}>{sample.name}</strong>
                      <small>{sample.media_type}</small>
                    </span>
                    <span className="sample-check" aria-hidden="true">✓</span>
                  </button>
                ))}
              </div>
            ) : (
              <p className="empty-samples">No sample media found in data/sample_data.</p>
            )}
          </div>

          {error && <p className="error-banner" role="alert">{error}</p>}
          {isProcessing && (
            <div className="analysis-progress" role="status" aria-live="polite">
              <div className="analysis-progress-heading">
                <strong>
                  {uploading
                    ? "Uploading footage"
                    : analysisJob?.phase === "loading_models"
                      ? "Loading detection models"
                      : analysisJob?.phase === "finalizing"
                        ? "Finalizing processed video"
                        : analysisJob?.status === "queued"
                          ? "Analysis queued"
                          : "Processing footage"}
                </strong>
                <span>
                  {uploading
                    ? "Upload in progress"
                    : analysisJob?.progress
                      ? `${analysisJob.progress}%`
                      : "Preparing"}
                </span>
              </div>
              <progress
                max="100"
                value={uploading ? 0 : analysisJob?.progress || 0}
                aria-label="Video analysis progress"
              />
              <div className="analysis-progress-details">
                <span>
                  {analysisJob?.current_frame && analysisJob?.total_frames
                    ? `Frame ${analysisJob.current_frame.toLocaleString()} of ${analysisJob.total_frames.toLocaleString()}`
                    : "Preparing video frames"}
                </span>
                <span>
                  {estimatedRemainingSeconds != null
                    ? `Estimated time remaining · ${formatDuration(estimatedRemainingSeconds)}`
                    : `Elapsed · ${formatDuration(elapsedSeconds)} · Estimating time`}
                </span>
              </div>
            </div>
          )}
          <div className="form-footer">
            <p>Each processed result is saved in its own analysis folder.</p>
            <button
              className="analyze-button"
              type="submit"
              disabled={isProcessing || (!upload && !selectedSample)}
            >
              {isProcessing ? (
                <><span className="spinner" /> {uploading ? "Uploading…" : "Processing…"}</>
              ) : (
                <>Analyze footage <span aria-hidden="true">→</span></>
              )}
            </button>
          </div>
        </form>
      </section>

      <section className="history-section" aria-label="Previous analyses">
        <details>
          <summary>
            <span>Previous analyses</span>
            <span className="history-count">
              {analysesError ? "UNAVAILABLE" : `${analyses.length} SAVED`}
            </span>
          </summary>
          {analysesError ? (
            <p className="inline-error">{analysesError}</p>
          ) : analyses.length ? (
            <div className="history-list">
              {analyses.map((analysis) => (
                <button
                  className={`history-item ${result?.analysis_id === analysis.analysis_id ? "selected" : ""}`}
                  key={analysis.analysis_id}
                  type="button"
                  onClick={() => showAnalysis(analysis)}
                  aria-pressed={result?.analysis_id === analysis.analysis_id}
                >
                  <span className="history-item-icon" aria-hidden="true">
                    {analysis.media_type === "video" ? "▶" : "▧"}
                  </span>
                  <span className="history-item-copy">
                    <strong title={analysis.input_name}>{analysis.input_name}</strong>
                    <small>
                      {analysis.media_type} ·{" "}
                      {new Date(analysis.created_at).toLocaleString()}
                    </small>
                  </span>
                  <span className="history-item-id">
                    {analysis.analysis_id.slice(0, 10)}
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <p className="empty-history">Completed analyses will appear here.</p>
          )}
        </details>
      </section>

      {result && (
        <section className="results-section" aria-labelledby="results-heading">
          <div className="section-heading">
            <div>
              <p className="eyebrow">ANALYSIS COMPLETE</p>
              <h2 id="results-heading">Processed footage</h2>
            </div>
            <span className="analysis-id">ID · {result.analysis_id.slice(0, 10)}</span>
          </div>
          <div className="results-grid">
            <div className="media-panel">
              <div className="panel-label">
                <span>ANNOTATED {result.media_type.toUpperCase()}</span>
                <a href={resultUrl} target="_blank" rel="noreferrer" download>
                  Download result ↓
                </a>
              </div>
              {result.media_type === "video" ? (
                <video key={resultUrl} src={resultUrl} controls preload="metadata" />
              ) : (
                <a className="result-image-link" href={resultUrl} target="_blank" rel="noreferrer">
                  <img src={resultUrl} alt="Processed image with detection annotations" />
                </a>
              )}
            </div>
            {result.operations.includes("ocr") && <aside className="ocr-panel">
              <div className="panel-label">
                <span>NUMBER PLATE OCR</span>
                <span className="ocr-count">{result.ocr_results.length}</span>
              </div>
              {result.ocr_results.length ? (
                <div className="ocr-table-wrap">
                  <table>
                    <thead><tr><th>VEHICLE ID</th><th>RECOGNIZED TEXT</th></tr></thead>
                    <tbody>
                      {result.ocr_results.map((entry, index) => (
                        <tr key={`${entry.vehicle_id}-${index}`}>
                          <td><span className="vehicle-tag">{entry.vehicle_id}</span></td>
                          <td>
                            <strong className="plate-text">{entry.text}</strong>
                            {entry.confidence != null && (
                              <small>{Math.round(entry.confidence * 100)}% confidence</small>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="ocr-empty">
                  <span aria-hidden="true">⌕</span>
                  <strong>No readable plates found</strong>
                  <p>Try a clearer image or video with visible plates.</p>
                </div>
              )}
            </aside>}
            {result.operations.includes("wrong_direction") && (
              <aside className="ocr-panel direction-panel">
                <div className="panel-label">
                  <span>WRONG-DIRECTION ALERTS</span>
                  <span className="ocr-count">
                    {result.wrong_direction_events?.length || 0}
                  </span>
                </div>
                {result.wrong_direction_events?.length ? (
                  <div className="direction-alert-list">
                    {result.wrong_direction_events.map((event, index) => (
                      <div
                        className="direction-alert"
                        key={`${event.vehicle_id}-${index}`}
                      >
                        <strong>{event.vehicle_id} · {event.vehicle_type}</strong>
                        <span>Against {formatDirection(event.direction)}</span>
                        <small>
                          {Math.round(event.confidence * 100)}% direction confidence
                        </small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="ocr-empty">
                    <strong>No confirmed wrong-direction vehicles</strong>
                    <p>
                      Lane: {result.lane_direction
                        ? formatDirection(result.lane_direction)
                        : "Not configured"}
                    </p>
                  </div>
                )}
              </aside>
            )}
            {result.operations.includes("red_light") && (
              <aside className="ocr-panel direction-panel">
                <div className="panel-label">
                  <span>RED-LIGHT VIOLATIONS</span>
                  <span className="ocr-count">
                    {result.red_light_events?.length || 0}
                  </span>
                </div>
                <div className="red-light-result-state">
                  Final signal state
                  <strong
                    className={`signal-state signal-${(result.red_light_signal_state || "UNKNOWN").toLowerCase()}`}
                  >
                    {result.red_light_signal_state || "UNKNOWN"}
                  </strong>
                </div>
                {result.red_light_events?.length ? (
                  <div className="direction-alert-list">
                    {result.red_light_events.map((event, index) => (
                      <div
                        className="direction-alert"
                        key={`${event.vehicle_id}-${event.frame}-${index}`}
                      >
                        <strong>{event.vehicle_id} · {event.vehicle_type}</strong>
                        <span>Crossed on {event.signal}</span>
                        <small>
                          Frame {event.frame.toLocaleString()}
                        </small>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="ocr-empty">
                    <strong>No confirmed red-light crossings</strong>
                    <p>
                      A violation is recorded only on a confirmed red signal
                      and configured crossing direction.
                    </p>
                  </div>
                )}
              </aside>
            )}
          </div>
        </section>
      )}

      <footer className="page-footer">
        <span>TRAFFIC VISION</span>
        <span>Local analysis · Results stay in the analysis folders</span>
      </footer>
    </main>
  );
}

export default App;
