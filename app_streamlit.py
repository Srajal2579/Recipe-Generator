import streamlit as st
import joblib
import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# --- 1. SET PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Recipe Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. FIX PYTHON PATHING ---
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

# --- 3. IMPORTS FROM BACKEND ---
try:
    from backend.utils.similarity import get_recommendations 
except ImportError as e:
    st.error(f"FATAL IMPORT ERROR: Could not find backend modules.")
    st.info(f"Current Working Directory: {os.getcwd()}")
    st.code(str(e))
    st.stop()

# --- 4. LOAD & CACHE ARTIFACTS ---
MODEL_PATH = "backend/model"

@st.cache_resource
def load_ml_artifacts():
    """Loads model artifacts once and caches them."""
    artifacts = {}
    
    # Init NLTK (Quietly)
    try:
        import nltk
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        nltk.download('stopwords', quiet=True)
    except Exception:
        pass 

    try:
        vec_path = os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl")
        mat_path = os.path.join(MODEL_PATH, "recipe_vectors.pkl")
        meta_path = os.path.join(MODEL_PATH, "recipes_metadata.pkl")

        if os.path.exists(vec_path) and os.path.exists(mat_path) and os.path.exists(meta_path):
            artifacts["vectorizer"] = joblib.load(vec_path)
            artifacts["tfidf_matrix"] = joblib.load(mat_path)
            artifacts["metadata"] = joblib.load(meta_path)
        else:
            artifacts["error"] = "Files missing"

    except Exception as e:
        artifacts["error"] = str(e)
        
    return artifacts

ARTIFACTS = load_ml_artifacts()

# --- 5. DISPLAY HELPER (UPDATED) ---
def display_recipe(recipe: Dict[str, Any], index: int):
    """Displays a recipe with Title, Image, and URL."""
    score = recipe.get('score', 0)
    score_pct = int(score * 100)
    
    # Create an expandable card
    with st.expander(f"{index}. {recipe['title']} ({score_pct}% Match)", expanded=(index==1)):
        col1, col2 = st.columns([1, 3])
        
        # COLUMN 1: Image
        with col1:
            img_url = recipe.get('image')
            # Check if image URL is valid
            if isinstance(img_url, str) and img_url.startswith('http'):
                # FIX: Changed 'use_container_width' to 'use_column_width' for compatibility
                st.image(img_url, use_column_width=True)
            else:
                st.markdown("🖼️ *No Image Available*")

        # COLUMN 2: Details
        with col2:
            st.markdown(f"**Cuisine Category:** {recipe['cuisine']}")
            
            # Link to full instructions
            url = recipe.get('url')
            if url and str(url) != 'nan':
                st.markdown(f"🔗 [**Click here for Full Instructions**]({url})")
            else:
                st.markdown("*Instructions not available online*")
            
            st.divider()
            st.markdown("#### Ingredients")
            ing_text = recipe.get('ingredients', 'No ingredients listed.')
            st.caption(ing_text)


# --- 6. MAIN APP UI ---

st.title("🥗 Chef's AI Recommender")
st.markdown("Tell us what's in your fridge, and we'll find a recipe from our database.")

# Check for Load Errors
if "error" in ARTIFACTS:
    if ARTIFACTS["error"] == "Files missing":
        st.error("🚨 Model files not found!")
        st.warning(f"Please move your .pkl files into: `{os.path.abspath(MODEL_PATH)}`")
    else:
        st.error(f"System Error: {ARTIFACTS['error']}")
    st.stop()

# Input Section
with st.container():
    col_input, col_btn = st.columns([4, 1])
    
    with col_input:
        user_input = st.text_input(
            "Your Ingredients:", 
            placeholder="e.g. chicken, apple, cinnamon, butter"
        )
    
    with col_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        # Note: If st.button fails too, remove 'use_container_width' there as well.
        submitted = st.button("Find Recipes", type="primary", use_container_width=True)

    top_n = st.slider("Results to show:", 1, 10, 5)

if submitted:
    if not user_input.strip():
        st.warning("Please enter some ingredients first!")
    else:
        with st.spinner(f"Searching for recipes with {user_input}..."):
            recommendations = get_recommendations(user_input, top_n, ARTIFACTS)
        
        if recommendations:
            st.success(f"Found {len(recommendations)} recipes!")
            for i, rec in enumerate(recommendations, 1):
                display_recipe(rec, i)
        else:
            st.warning("No close matches found. Try using simpler ingredient names.")