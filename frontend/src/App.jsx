
import { useEffect, useState } from "react";
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
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

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

    fetch(`${API_BASE}/api/wrong-direction/config`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error("Could not load wrong-direction settings.");
        }
        return response.json();
      })
      .then((config) => {
        if (!active) return;
        setLaneDirections(config.directions);
        setLaneDirection(config.default_direction);
        setLanePolygon(config.lane_polygon.map(([x, y]) => [x * 100, y * 100]));
      })
      .catch((fetchError) => {
        if (active) setLaneConfigError(fetchError.message);
      });

    return () => {
      active = false;
    };
  }, []);

  function chooseSample(name) {
    setSelectedSample(name);
    setUpload(null);
    setResult(null);
    setError("");
  }

  function chooseUpload(event) {
    const nextFile = event.target.files?.[0] || null;
    event.target.value = "";
    setUpload(nextFile);
    setSelectedSample("");
    setResult(null);
    setError("");
  }

  function dropUpload(event) {
    event.preventDefault();
    const nextFile = event.dataTransfer.files?.[0] || null;
    if (!nextFile) return;
    setUpload(nextFile);
    setSelectedSample("");
    setResult(null);
    setError("");
  }

  function toggleOperation(operation, checked) {
    const next = new Set(selectedOperations);
    if (checked) {
      next.add(operation);
      if (
        operation === "helmet" ||
        operation === "plate" ||
        operation === "ocr" ||
        operation === "wrong_direction"
      ) {
        next.add("vehicle");
      }
      if (operation === "ocr") next.add("plate");
    } else {
      next.delete(operation);
      if (operation === "vehicle") {
        next.delete("helmet");
        next.delete("plate");
        next.delete("ocr");
        next.delete("wrong_direction");
      }
      if (operation === "plate") next.delete("ocr");
    }
    setSelectedOperations(next);
    setResult(null);
    setError("");
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
    if (selectedOperations.has("wrong_direction")) {
      body.append("lane_direction", laneDirection);
      body.append(
        "lane_polygon",
        JSON.stringify(
          lanePolygon.map(([x, y]) => [Number(x) / 100, Number(y) / 100]),
        ),
      );
    }

    setIsProcessing(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Analysis failed.");
      }
      setResult(payload);
      setAnalyses((previous) => [
        payload,
        ...previous.filter((item) => item.analysis_id !== payload.analysis_id),
      ]);
    } catch (requestError) {
      setError(requestError.message || "Could not connect to the analysis server.");
    } finally {
      setIsProcessing(false);
    }
  }

  const resultUrl = result
    ? new URL(result.result_url, API_BASE).toString()
    : "";

  function showAnalysis(analysis) {
    setResult(analysis);
    setError("");
  }

  function updateLanePoint(pointIndex, coordinateIndex, value) {
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
                    selectedOperations.has("wrong_direction"));
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
                    onChange={(event) => setLaneDirection(event.target.value)}
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
                      {laneConfigError} Using the default lane polygon.
                    </p>
                  )}
                </fieldset>
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
          <div className="form-footer">
            <p>Each processed result is saved in its own analysis folder.</p>
            <button
              className="analyze-button"
              type="submit"
              disabled={isProcessing || (!upload && !selectedSample)}
            >
              {isProcessing ? (
                <><span className="spinner" /> Processing footage…</>
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
