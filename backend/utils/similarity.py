import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

# --- 1. NLP Setup ---
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

# Define Stop Words
english_stop_words = set(stopwords.words('english'))
custom_recipe_stop_words = {
    'fresh', 'dry', 'frozen', 'cup', 'ounce', 'pound', 'tablespoon', 'teaspoon', 
    'clove', 'package', 'can', 'jar', 'diced', 'chopped', 'minced', 'sliced', 
    'pinch', 'large', 'small', 'medium', 'container', 'dash', 'serving', 'to', 
    'oz', 'tbsp', 'tsp', 'g', 'kg', 'ml', 'l', 'of', 'and', 'with', 'or', 'taste'
}
ALL_STOP_WORDS = english_stop_words.union(custom_recipe_stop_words)


# --- 2. Preprocessing Function ---
def clean_ingredient_text(text: str) -> str:
    """
    Cleans a string of ingredients (from recipes.csv) into a normalized string 
    for the vectorizer.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase & remove non-alphabet characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize, Lemmatize, and remove Stop Words
    words = []
    for w in text.split():
        if w not in ALL_STOP_WORDS and len(w) > 2:
            words.append(lemmatizer.lemmatize(w))
    
    return " ".join(words)


# --- 3. Recommendation Logic ---
def get_recommendations(
    user_ingredients: str, 
    top_n: int, 
    artifacts: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Generates recommendations using the new recipes.csv metadata structure.
    """
    
    vectorizer = artifacts.get("vectorizer")
    tfidf_matrix = artifacts.get("tfidf_matrix")
    metadata = artifacts.get("metadata")

    # ERROR PREVENTION: Check if artifacts are missing (Fixed 'ambiguous truth value' error)
    if vectorizer is None or tfidf_matrix is None or metadata is None:
        print("Error: Missing required model artifacts.")
        return []

    # 1. Preprocess User Input
    # User input is a string "chicken, garlic". We clean it directly.
    user_clean_text = clean_ingredient_text(user_ingredients)
    
    # 2. Vectorize Input
    user_vec = vectorizer.transform([user_clean_text])
    
    # 3. Calculate Similarity
    # 
    cos_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # 4. Get Top N Results
    # Sort by score descending
    top_indices = np.argsort(cos_sim)[::-1][:top_n]
    
    recs = []
    for idx in top_indices:
        score = float(cos_sim[idx])
        
        # Threshold: Ignore recipes with very low similarity (noise)
        if score > 0.01: 
            recipe_data = metadata[idx]
            
            # Construct the recipe object using the NEW keys (title, url, etc.)
            rec = {
                "title": recipe_data.get("title", "Unknown Recipe"),
                "cuisine": recipe_data.get("cuisine", "General"),
                "ingredients": recipe_data.get("ingredients", ""),
                "url": recipe_data.get("url", ""),
                "image": recipe_data.get("img_src", ""),
                "score": score
            }
            recs.append(rec)
        
    return recs