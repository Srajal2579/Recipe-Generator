import joblib
import pandas as pd
import os
from contextlib import asynccontextmanager # New import for lifespan
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Import Core Logic ---
# Ensure you run this from the root directory using: uvicorn backend.main:app --reload
from backend.utils.similarity import get_recommendations
# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data")

# --- Schemas ---
class RecommendationRequest(BaseModel):
    ingredients: str
    top_n: int = 5

class Recipe(BaseModel):
    id: int
    cuisine: str
    ingredients: List[str]
    score: float

# --- Global Storage ---
MODEL_ARTIFACTS = {}

# --- Lifespan Manager (Replaces @app.on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown logic.
    1. Loads ML models when the app starts.
    2. Cleans up when the app stops.
    """
    # --- STARTUP LOGIC ---
    try:
        print("Loading ML artifacts...")
        
        # Define paths
        vec_path = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(MODEL_PATH, "recipe_vectors.pkl")
        meta_path = os.path.join(MODEL_PATH, "recipes_metadata.pkl")

        # Load files
        if os.path.exists(vec_path) and os.path.exists(mat_path) and os.path.exists(meta_path):
            MODEL_ARTIFACTS["vectorizer"] = joblib.load(vec_path)
            MODEL_ARTIFACTS["tfidf_matrix"] = joblib.load(mat_path)
            MODEL_ARTIFACTS["metadata"] = joblib.load(meta_path)
            print("✅ Model artifacts loaded successfully!")
        else:
            print(f"⚠️ Warning: Artifacts not found in {MODEL_PATH}. API will return errors.")
            MODEL_ARTIFACTS["error"] = "Artifacts missing"

    except Exception as e:
        print(f"❌ CRITICAL ERROR loading models: {e}")
        MODEL_ARTIFACTS["error"] = str(e)

    # Yield control back to the application (Server starts running here)
    yield
    
    # --- SHUTDOWN LOGIC ---
    print("Cleaning up resources...")
    MODEL_ARTIFACTS.clear()


# --- App Init ---
app = FastAPI(
    title="Recipe Recommender API", 
    description="ML-powered recipe suggestions.",
    lifespan=lifespan  # Pass the lifespan manager here
)


# --- Endpoints ---
@app.get("/")
def home():
    return {"message": "Recipe Recommender API is running!", "status": "active"}

@app.post("/api/recommend", response_model=List[Recipe])
async def recommend(request: RecommendationRequest):
    # Check if models loaded correctly
    if "error" in MODEL_ARTIFACTS:
        raise HTTPException(status_code=500, detail="Model artifacts failed to load.")
    
    if not MODEL_ARTIFACTS:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    # Get recommendations
    recommendations = get_recommendations(
        request.ingredients,
        request.top_n,
        artifacts=MODEL_ARTIFACTS
    )
    
    return recommendations