import React, { useState } from "react";
import ForecastCard from "./components/ForecastCard";
import "./styles.css";

function App() {
  const [location, setLocation] = useState("");
  const [forecast, setForecast] = useState(null);

  const fetchForecast = async (horizon="current") => {
    const res = await fetch(`http://localhost:8000/predict?location=${location}&horizon=${horizon}`);
    const data = await res.json();
    setForecast(data);
  };

  return (
    <div className="app">
      <h1>AI Weather Forecast</h1>
      <input
        type="text"
        placeholder="Enter location..."
        value={location}
        onChange={(e) => setLocation(e.target.value)}
      />
      <div className="buttons">
        <button onClick={() => fetchForecast("current")}>Current</button>
        <button onClick={() => fetchForecast("next6h")}>Next 6h</button>
        <button onClick={() => fetchForecast("nextday")}>Next Day</button>
      </div>
      {forecast && <ForecastCard forecast={forecast} />}
    </div>
  );
}

export default App;