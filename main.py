import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

BASE = Path(__file__).parent
model = joblib.load(BASE / "knn_heart_model.pkl")
scaler = joblib.load(BASE / "heart_scaler.pkl")
COLS = list(joblib.load(BASE / "heart_columns.pkl"))

# Column of predict_proba that belongs to "heart disease" (class 1), found
# explicitly instead of assuming it is always index 1.
POSITIVE = list(model.classes_).index(1)

NUMERIC = ("Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR")
CATEGORICAL = ("Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope")
MIN_DROP = 0.01  # ignore changes smaller than 1 percentage point

app = FastAPI(title="Heart Disease Prediction API")

# Only needed if the page is opened from another origin (e.g. Live Server).
# Example: ALLOWED_ORIGINS="http://127.0.0.1:5500,http://localhost:5500"
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins,
                       allow_methods=["POST", "GET"], allow_headers=["*"])


class Patient(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject typos like "Cholestrol"

    Age: int = Field(ge=18, le=100)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["ATA", "NAP", "TA", "ASY"]
    RestingBP: int = Field(ge=80, le=200)
    Cholesterol: int = Field(ge=100, le=600)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: int = Field(ge=60, le=220)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=0, le=6)
    ST_Slope: Literal["Up", "Flat", "Down"]


class Lever(BaseModel):
    label: str
    drop: float          # how much the probability falls (0.12 = 12 points)
    actionable: bool     # False = a test finding, not something you can change directly


class Prediction(BaseModel):
    probability: float
    prediction: int
    risk_band: Literal["low", "moderate", "high"]
    levers: list[Lever]


def to_row(p: dict) -> dict:
    row = {c: 0 for c in COLS}
    for k in NUMERIC:
        row[k] = p[k]
    row["Oldpeak"] = int(p["Oldpeak"])  # notebook cast all columns to int before training
    for k in CATEGORICAL:
        col = f"{k}_{p[k]}"
        if col in row:  # reference categories (F, ASY, LVH, N, Down) have no column
            row[col] = 1
    return row


def probabilities(patients: list[dict]) -> list[float]:
    """Score many patients in ONE scaler + KNN call (much faster than one call each)."""
    X = scaler.transform(pd.DataFrame([to_row(p) for p in patients], columns=COLS))
    return model.predict_proba(X)[:, POSITIVE].tolist()


# What-if scenarios: (label, change, applies_when, actionable).
# A scenario only runs when it would actually be an improvement for this patient,
# so someone with cholesterol 150 is never told to "bring it down to 200".
SCENARIOS = [
    ("Bring cholesterol down to 200", {"Cholesterol": 200}, lambda p: p["Cholesterol"] > 200, True),
    ("Bring blood pressure down to 120", {"RestingBP": 120}, lambda p: p["RestingBP"] > 120, True),
    ("Get fasting blood sugar back to normal", {"FastingBS": 0}, lambda p: p["FastingBS"] == 1, True),
    ("Raise max heart rate to 160", {"MaxHR": 160}, lambda p: p["MaxHR"] < 160, True),
    ("No exercise-induced angina", {"ExerciseAngina": "N"}, lambda p: p["ExerciseAngina"] == "Y", False),
    ("No ST depression (Oldpeak 0)", {"Oldpeak": 0}, lambda p: int(p["Oldpeak"]) > 0, False),
    ("Upward ST slope", {"ST_Slope": "Up"}, lambda p: p["ST_Slope"] != "Up", False),
]


def band(prob: float) -> str:
    return "low" if prob < 0.30 else "moderate" if prob < 0.60 else "high"


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    p = patient.model_dump()
    todo = [(label, change, act) for label, change, applies, act in SCENARIOS if applies(p)]

    combined: dict = {}
    for _, change, _ in todo:
        combined.update(change)

    # One batch: [current, each scenario..., all scenarios together]
    batch = [p] + [{**p, **c} for _, c, _ in todo]
    if todo:
        batch.append({**p, **combined})
    probs = probabilities(batch)
    base = probs[0]

    levers = [
        {"label": label, "drop": round(base - pr, 3), "actionable": act}
        for (label, _, act), pr in zip(todo, probs[1:])
        if base - pr >= MIN_DROP
    ]
    levers.sort(key=lambda x: -x["drop"])
    levers = levers[:3]

    if todo and len(levers) > 1:
        combo = base - probs[-1]
        if combo >= MIN_DROP:
            levers.insert(0, {"label": "All of these together", "drop": round(combo, 3),
                              "actionable": all(l["actionable"] for l in levers)})

    return {
        "probability": round(base, 3),
        "prediction": int(base >= 0.5),
        "risk_band": band(base),
        "levers": levers,
    }


@app.get("/health")
def health():
    return {"status": "ok", "features": len(COLS)}


@app.get("/")
def home():
    return FileResponse(BASE / "index.html")
