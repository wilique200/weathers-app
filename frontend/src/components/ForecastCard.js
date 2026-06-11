import React from "react";

function ForecastCard({ forecast }) {
  return (
    <div className="card">
      <h2>{forecast.location}</h2>
      <p>Period: {forecast.forecast_period}</p>
      <p>Rain: {forecast.rainfall_prediction} ({forecast.rain_probability})</p>
      <p>Temperature: {forecast.predicted_temperature_C} °C</p>
      <p>Wind: {forecast.predicted_wind_speed_m_s} m/s</p>
    </div>
  );
}

export default ForecastCard;