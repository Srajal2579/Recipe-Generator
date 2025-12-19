# 🍳 AI Recipe Recommender

A Full Stack application that recommends recipes based on ingredients using TF-IDF and Cosine Similarity.

## Project Structure
- **Backend:** FastAPI (Python)
- **Frontend:** React + TailwindCSS
- **ML:** Scikit-Learn (TF-IDF)

## Setup Instructions

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
# Ensure you have run the training notebook to generate model files in backend/model/
uvicorn main:app --reload