from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import PredictRequest, PredictResponse, HealthResponse
from src.api.predictor import predictor

app = FastAPI(
    title="DealPulse AI",
    description="Smart Opportunity Stage Predictor",
    version="1.0.0"
)

# Allow UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Store recent predictions in memory
recent_predictions = []


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status":  "ok",
        "model":   "distilbert-base-uncased",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predictor.predict(request.deal_id, request.crm_notes)

        # Save to recent predictions
        recent_predictions.append(result)
        if len(recent_predictions) > 50:
            recent_predictions.pop(0)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recent")
def get_recent():
    return {"predictions": recent_predictions[-10:]}


@app.get("/stages")
def get_stages():
    return {
        "stages": [
            {"id": 0, "name": "Prospecting",  "color": "#6366f1"},
            {"id": 1, "name": "Qualification","color": "#f59e0b"},
            {"id": 2, "name": "Proposal",     "color": "#3b82f6"},
            {"id": 3, "name": "Negotiation",  "color": "#8b5cf6"},
            {"id": 4, "name": "Closed Won",   "color": "#10b981"},
            {"id": 5, "name": "Closed Lost",  "color": "#ef4444"}
        ]
    }