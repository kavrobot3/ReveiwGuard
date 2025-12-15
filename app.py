from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import re
import html
import joblib

# Initialize FastAPI app
app = FastAPI(
    title="AI Review Detection API",
    description="Predict if a review is AI-generated and estimate regret score",
    version="1.0.0"
)

# Load models at startup
ai_detector = joblib.load('ai_detector.pkl')
regret_predictor = joblib.load('regret_predictor.pkl')

# Request model
class ReviewRequest(BaseModel):
    text: str = Field(..., description="Review text to analyze", min_length=1)
    rating: float = Field(default=3.0, description="Rating 1-5", ge=1.0, le=5.0)

# Response model
class ReviewResponse(BaseModel):
    is_ai_generated: bool
    ai_confidence_percent: float
    regret_score_1_10: float
    verdict: str

def clean_text(text_input):
    """Clean text by removing HTML tags, emojis, special chars"""
    if not isinstance(text_input, str) or not text_input.strip():
        return text_input
    
    cleaned = html.unescape(text_input)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    cleaned = emoji_pattern.sub(r'', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?\'\"-]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def extract_features(text, rating):
    """Extract all 9 features from text and rating"""
    cleaned_text = clean_text(text)
    
    # Feature 1: length
    length = len(cleaned_text)
    
    # Feature 2: word_count
    word_count = len(cleaned_text.split())
    
    # Feature 3: simple_sentiment
    positive_keywords = ['great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'good', 'awesome', 'happy', 'recommend']
    negative_keywords = ['bad', 'worst', 'terrible', 'hate', 'poor', 'awful', 'disappointed', 'waste', 'broken', 'useless']
    
    text_lower = cleaned_text.lower()
    pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    simple_sentiment = pos_count - neg_count
    
    # Feature 4: rating_sentiment_diff
    rating_sentiment_diff = rating - (simple_sentiment + 3)
    
    # Feature 5: exclamation_count
    exclamation_count = cleaned_text.count('!')
    
    # Feature 6: question_count
    question_count = cleaned_text.count('?')
    
    # Feature 7: caps_ratio
    letters = [c for c in cleaned_text if c.isalpha()]
    caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters) if letters else 0.0
    
    # Feature 8: i_count
    i_count = len(re.findall(r'\bI\b', cleaned_text))
    
    return [rating, length, word_count, simple_sentiment, rating_sentiment_diff, 
            exclamation_count, question_count, caps_ratio, i_count]

@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "AI Review Detection API is running"}

@app.post("/predict", response_model=ReviewResponse, tags=["Prediction"])
def predict(request: ReviewRequest):
    """Predict if a review is AI-generated and estimate regret score"""
    try:
        # Extract features
        features = extract_features(request.text, request.rating)
        
        # Create DataFrame with correct feature names
        features_df = pd.DataFrame([features], columns=[
            'rating', 'length', 'word_count', 'simple_sentiment',
            'rating_sentiment_diff', 'exclamation_count', 'question_count',
            'caps_ratio', 'i_count'
        ])
        
        features_df = features_df.fillna(0)
        
        # Predictions
        is_ai = int(ai_detector.predict(features_df)[0])
        ai_proba = ai_detector.predict_proba(features_df)[0]
        ai_confidence = float(ai_proba[1] * 100)
        
        regret_score = float(regret_predictor.predict(features_df)[0])
        regret_score = max(1.0, min(10.0, regret_score))
        
        # Generate verdict
        if is_ai == 1:
            verdict = f"AI-generated ({ai_confidence:.1f}% confidence)"
        else:
            verdict = f"Human-written ({100-ai_confidence:.1f}% confidence)"
        
        return ReviewResponse(
            is_ai_generated=bool(is_ai),
            ai_confidence_percent=round(ai_confidence, 2),
            regret_score_1_10=round(regret_score, 2),
            verdict=verdict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
