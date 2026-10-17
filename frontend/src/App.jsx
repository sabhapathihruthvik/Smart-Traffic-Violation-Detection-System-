
import { useEffect, useState } from "react";

function App() {
  const [vehicles, setVehicles] = useState([]);
  const [violations, setViolations] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadData() {
      try {
        const [vehiclesRes, violationsRes] = await Promise.all([
          fetch("http://127.0.0.1:8000/api/vehicles"),
          fetch("http://127.0.0.1:8000/api/violations"),
        ]);

        if (!vehiclesRes.ok || !violationsRes.ok) {
          throw new Error("Failed to fetch backend data");
        }

        setVehicles(await vehiclesRes.json());
        setViolations(await violationsRes.json());
      } catch (err) {
        setError(err.message);
      }
    }

    loadData();
  }, []);

  return (
    <main style={{ padding: "24px" }}>
      <h1>Smart Traffic Violation Detection</h1>

      {error && <p>{error}. Is the Python backend running?</p>}

      <h2>Vehicles</h2>
      {vehicles.map((v) => (
        <p key={v.id}>
          {v.vehicle_number} — {v.vehicle_type}
        </p>
      ))}

      <h2>Violations</h2>
      {violations.map((v) => (
        <p key={v.id}>
          {v.vehicle_number || "Unknown vehicle"} — {v.violation_type}
          {" — "}{v.verification_status}
        </p>
      ))}
    </main>
  );
}

export default App;
