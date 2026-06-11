import json, requests, pandas as pd, numpy as np, torch, joblib

# --- Load features list ---
with open("backend/models/features.json", "r") as f:
    features = json.load(f)
input_dim = len(features)

# --- Load scaler ---
scaler = joblib.load("backend/models/scaler.pkl")

# --- Define DL models ---
class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 2)
        )
    def forward(self, x): return self.net(x)

class MLPRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# --- Load trained weights ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rain_model = MLPClassifier(input_dim).to(device)
temp_model = MLPRegressor(input_dim).to(device)
wind_model = MLPRegressor(input_dim).to(device)

rain_model.load_state_dict(torch.load("backend/models/dl_rain.pt", map_location=device))
temp_model.load_state_dict(torch.load("backend/models/dl_temp.pt", map_location=device))
wind_model.load_state_dict(torch.load("backend/models/dl_wind.pt", map_location=device))

rain_model.eval(); temp_model.eval(); wind_model.eval()

# --- Feature engineering ---
def engineer_features(df):
    df["month"] = df["date"].dt.month
    df["hour"] = df["date"].dt.hour
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    T = df["temperature_2m"]; RH = df["relative_humidity_2m"]
    df["heat_index"] = (-8.784695 + 1.61139411*T + 2.338549*RH
        - 0.14611605*T*RH - 0.012308094*(T**2)
        - 0.016424828*(RH**2) + 0.002211732*(T**2)*RH
        + 0.00072546*T*(RH**2) - 0.000003582*(T**2)*(RH**2))
    es = 0.6108*np.exp((17.27*T)/(T+237.3)); ea = es*(RH/100)
    df["vapour_pressure_deficit_calc"] = es - ea
    df["temp_humidity_interaction"] = T*RH
    df["wind_power_potential"] = 0.5*1.225*(df["wind_speed_10m"]**3)
    df["log_precipitation"] = np.log1p(df["precipitation"])
    df.drop(columns=["month","hour"], inplace=True)
    return df

# --- Geocoding ---
def geocode_location(location_name):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location_name, "format": "json", "limit": 1}
    resp = requests.get(url, params=params, headers={"User-Agent": "weather-app"})
    resp.raise_for_status()
    data = resp.json()
    if not data:
        fallback = " ".join(location_name.split()[:-1])
        if fallback:
            params["q"] = fallback
            resp = requests.get(url, params=params, headers={"User-Agent": "weather-app"})
            data = resp.json()
    if not data:
        raise ValueError(f"Could not geocode location: {location_name}")
    lat = float(data[0]["lat"]); lon = float(data[0]["lon"])
    return lat, lon

# --- Fetch features ---
def fetch_features(location_name):
    lat, lon = geocode_location(location_name)
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&hourly="
        "temperature_2m,relative_humidity_2m,dew_point_2m,"
        "apparent_temperature,cloud_cover,pressure_msl,"
        "wind_speed_10m,wind_direction_10m,shortwave_radiation,"
        "soil_temperature_0_to_7cm,soil_moisture_0_to_7cm,"
        "precipitation,rain"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data["hourly"])
    df["date"] = pd.to_datetime(df["time"])
    return df

# --- Inference ---
def predict_weather(location_name, horizon="current"):
    df = fetch_features(location_name)
    df = engineer_features(df)

    if horizon == "current":
        df_sel = df.tail(1)
    elif horizon == "next6h":
        df_sel = df.tail(6)
    elif horizon == "nextday":
        df_sel = df.iloc[-24:]
    else:
        raise ValueError("Invalid horizon. Use 'current', 'next6h', or 'nextday'.")

    X = df_sel[features].values
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        dl_logits = rain_model(torch.tensor(X_scaled, dtype=torch.float32).to(device))
        rain_prob = torch.softmax(dl_logits, dim=1)[:,1].cpu().numpy()
        temp_pred = temp_model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()
        wind_pred = wind_model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy()

    rain_label = "Rain" if np.mean(rain_prob) >= 0.5 else "No Rain"
    return {
        "location": location_name,
        "forecast_period": horizon,
        "rainfall_prediction": rain_label,
        "rain_probability": round(float(np.mean(rain_prob)), 3),
        "predicted_temperature_C": round(float(np.mean(temp_pred)), 2),
        "predicted_wind_speed_m_s": round(float(np.mean(wind_pred)), 2)
    }