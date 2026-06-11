from fastapi import FastAPI, Query
from inference_pipeline import predict_weather

app = FastAPI(title="AI Weather Forecast")

@app.get("/predict")
def get_forecast(location: str = Query(...), horizon: str = Query("current")):
    """
    Example:
    /predict?location=Osara,Kogi%20State,Nigeria&horizon=nextday
    """
    return predict_weather(location, horizon)