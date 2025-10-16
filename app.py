from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --------------------------
# Initialize FastAPI
# --------------------------
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

# --------------------------
# Load Model, Scaler & Data
# --------------------------
lstm_model = load_model("lstm_close_model.h5", compile=False)
scaler = joblib.load("lstm_scaler.pkl")

df = pd.read_csv("synthetic_market.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)


# --------------------------
# Forecast Function
# --------------------------
def forecast_next_days_lstm(df, n_days=5, n_steps=10):
    close_prices = df["close"].values
    forecasted = []
    last_sequence = close_prices[-n_steps:]
    
    for day in range(1, n_days + 1):
        seq_scaled = scaler.transform(last_sequence.reshape(-1, 1)).reshape(1, n_steps, 1)
        pred_scaled = lstm_model.predict(seq_scaled, verbose=0)
        pred_close = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

        last_date = df["date"].max()
        forecast_date = last_date + pd.Timedelta(days=day)
        forecasted.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "pred_close": float(pred_close)
        })
        last_sequence = np.append(last_sequence[1:], pred_close)

    return forecasted


# --------------------------
# Routes
# --------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/BSE", response_class=HTMLResponse)
def bse_home(request: Request):
    return templates.TemplateResponse("BSE.html", {"request": request, "forecast": None})


@app.post("/forecast_bse", response_class=HTMLResponse)
def forecast_bse(request: Request, days: int = Form(...)):
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("BSE.html", {"request": request, "forecast": forecasted})


@app.route("/forecast_bse", methods=["GET", "POST"], response_class=HTMLResponse)
async def forecast_bse(request: Request, days: int = Form(None)):
    if request.method == "GET":
        return templates.TemplateResponse("BSE.html", {"request": request, "forecast": None})
    
    forecasted = forecast_next_days_lstm(df, n_days=days)
    return templates.TemplateResponse("BSE.html", {"request": request, "forecast": forecasted})
