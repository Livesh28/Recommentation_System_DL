import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from recommender import RecommenderEngine

app = FastAPI(
    title="EcoPulse AI - Sustainable Recommendation System",
    description="Backend API for Semantic Search, Sustainability Prediction, and Personalized Product Recommendation",
    version="1.0.0"
)

# Initialize recommender engine (assumes bootstrap.py has run successfully)
engine = RecommenderEngine()

# Ensure static folder exists and mount it
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')
os.makedirs(static_dir, exist_ok=True)

# ----------------- REQUEST MODELS -----------------

class InteractionRequest(BaseModel):
    user_id: str = Field(..., example="user_15")
    product_id: str = Field(..., example="25")
    interaction_score: int = Field(..., ge=1, le=5, example=5)

class PredictionRequest(BaseModel):
    category: str = Field(..., example="Apparel")
    material: str = Field(..., example="Organic Cotton")
    packaging_type: str = Field(..., example="Compostable Bag")
    certifications: str = Field(..., example="GOTS")
    manufacturer_country: str = Field(..., example="Portugal")
    carbon_footprint_g: int = Field(..., ge=0, example=950)
    water_usage_L: int = Field(..., ge=0, example=1200)
    recycled_content_pct: int = Field(..., ge=0, le=100, example=90)
    price_usd: float = Field(..., ge=0.0, example=45.0)

class Weights(BaseModel):
    semantic_similarity: float = Field(0.40, ge=0.0, le=1.0)
    personalized_ranking: float = Field(0.30, ge=0.0, le=1.0)
    sustainability_score: float = Field(0.30, ge=0.0, le=1.0)

class HybridRequest(BaseModel):
    query: str = Field(..., example="water bottle")
    user_id: Optional[str] = Field(None, example="user_1")
    limit: int = Field(10, ge=1, le=50)
    weights: Optional[Weights] = None

# ----------------- UI INDEX ROUTE -----------------

@app.get("/")
def read_root():
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to EcoPulse API! Create 'static/index.html' to access the frontend."}

# ----------------- API ENDPOINTS -----------------

@app.get("/api/users")
def get_users():
    """Returns a list of active users from the interactions database."""
    try:
        engine.load_interactions()
        # Find unique users with interactions
        unique_users = engine.interactions_df['user_id'].unique().tolist()
        # Sort users numerically if they follow user_X pattern
        try:
            unique_users.sort(key=lambda x: int(x.split('_')[1]) if '_' in x else 9999)
        except Exception:
            unique_users.sort()
        return {"users": unique_users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
def get_filters():
    """Returns unique categories and materials for the UI dropdowns."""
    try:
        categories = engine.products_df['category'].unique().tolist()
        materials = engine.products_df['material'].unique().tolist()
        certifications = engine.products_df['certifications'].unique().tolist()
        
        categories.sort()
        materials.sort()
        certifications = [c for c in certifications if c != 'None']
        certifications.sort()
        
        return {
            "categories": categories,
            "materials": materials,
            "certifications": certifications
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products")
def get_products(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    category: Optional[str] = None,
    material: Optional[str] = None,
    min_eco_score: Optional[float] = None,
    search: Optional[str] = None
):
    """Retrieves paginated and filtered products, along with dashboard aggregates."""
    try:
        df = engine.products_df.copy()
        
        # Apply filters
        if category and category != "All":
            df = df[df['category'] == category]
        if material and material != "All":
            df = df[df['material'] == material]
        if min_eco_score is not None:
            df = df[df['eco_score'] >= min_eco_score]
        if search:
            df = df[df['product_name'].str.contains(search, case=False) | 
                    df['material'].str.contains(search, case=False)]
            
        total_count = len(df)
        
        # Computes Dashboard Analytics based on the full (filtered) dataset
        avg_sustainability = float(df['sustainability_index_normalized'].mean()) if total_count > 0 else 0.0
        avg_carbon = float(df['carbon_footprint_g'].mean()) if total_count > 0 else 0.0
        avg_water = float(df['water_usage_L'].mean()) if total_count > 0 else 0.0
        avg_price = float(df['raw_price_usd'].mean()) if total_count > 0 else 0.0
        
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated_df = df.iloc[start:end]
        
        products_list = paginated_df.to_dict(orient='records')
        
        return {
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "products": products_list,
            "stats": {
                "avg_sustainability_index": round(avg_sustainability, 2),
                "avg_carbon_footprint_g": round(avg_carbon, 1),
                "avg_water_usage_l": round(avg_water, 1),
                "avg_price_usd": round(avg_price, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search")
def search_products(query: str, limit: int = Query(5, ge=1, le=50)):
    """Performs semantic search via SentenceTransformer embeddings + FAISS index."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    try:
        results = engine.search_products(query, k=limit)
        return results.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/user")
def recommend_for_user(user_id: str, limit: int = Query(10, ge=1, le=50)):
    """Retrieves user-personalized recommendations based on user interest embedding profile."""
    try:
        results = engine.recommend_for_user(user_id, k=limit)
        return results.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommend/hybrid")
def recommend_hybrid(request: HybridRequest):
    """Retrieves personalized recommendations matching a query, fusing scores with custom weights."""
    try:
        weights_dict = None
        if request.weights:
            weights_dict = {
                "semantic_similarity": request.weights.semantic_similarity,
                "personalized_ranking": request.weights.personalized_ranking,
                "sustainability_score": request.weights.sustainability_score
            }
            # Re-normalize weights to sum to 1.0
            w_sum = sum(weights_dict.values())
            if w_sum > 0:
                weights_dict = {k: v / w_sum for k, v in weights_dict.items()}
                
        results = engine.hybrid_recommend_by_nlq(
            query=request.query,
            user_id=request.user_id,
            k=request.limit,
            weights=weights_dict
        )
        return results.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommend/similar")
def recommend_similar(product_id: str, limit: int = Query(5, ge=1, le=50)):
    """Retrieves list of similar products to a given product_id based on FAISS nearest neighbors."""
    try:
        results = engine.recommend_similar(product_id, k=limit)
        if results.empty:
            raise HTTPException(status_code=404, detail="Product ID not found or no similar products found.")
        return results.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interact")
def record_interaction(request: InteractionRequest):
    """Logs a user interaction (clicks, views, purchases) to dynamically update recommendation profiles."""
    try:
        # Verify product exists
        if not (engine.products_df['product_id'] == request.product_id).any():
            raise HTTPException(status_code=404, detail=f"Product with ID '{request.product_id}' not found.")
            
        engine.add_interaction(request.user_id, request.product_id, request.interaction_score)
        return {"status": "success", "message": f"Recorded interaction for user {request.user_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict")
def predict_sustainability(request: PredictionRequest):
    """Predicts sustainability metrics for custom product specifications using the trained ML model."""
    try:
        pred_data = request.model_dump()
        # The model uses scaled price, predict handles that scaling using max/min of products_df
        predictions = engine.predict_sustainability(pred_data)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/explain")
def explain_recommendation(
    product_id: str,
    query: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Generates structured reasoning explanations for why a product was recommended."""
    try:
        p_row = engine.products_df[engine.products_df['product_id'] == product_id]
        if p_row.empty:
            raise HTTPException(status_code=404, detail="Product ID not found.")
            
        reasons = engine.generate_explanation(p_row.iloc[0], query=query, user_id=user_id)
        return {"reasons": reasons}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files as final fallback
app.mount("/", StaticFiles(directory=static_dir), name="static_root")
