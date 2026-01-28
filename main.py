from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("oracle_model.pkl")

class MatchData(BaseModel):
    tier: int
    bo: int
    t1_wr: float
    t2_wr: float

@app.post("/predict")
def predict(data: MatchData):
    X = pd.DataFrame([data.dict()])
    probs = model.predict_proba(X)[0]
    return {
        "t1_prob": round(probs[1] * 100, 2),
        "t2_prob": round(probs[0] * 100, 2),
        "verdict": "team1" if probs[1] > 0.5 else "team2"
    }