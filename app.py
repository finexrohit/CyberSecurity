"""
Cyberbullying Detection Using Machine Learning
Flask Web Application

This application provides a web interface for detecting cyberbullying
in text using Natural Language Processing and Machine Learning.

Author: College Major Project
Date: December 2025
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import os
import numpy as np

# NLP Libraries
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except ImportError:
    print("NLTK not installed. Run: pip install nltk")

# Initialize Flask application
app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None

def load_model():
    """
    Load the trained machine learning model and TF-IDF vectorizer.
    Returns True if successful, False otherwise.
    """
    global model, vectorizer
    
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
    
    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print("Model and vectorizer loaded successfully!")
            return True
        else:
            print("Model files not found. Please run train_model.py first.")
            return False
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_text(text):
    """
    Preprocess the input text using NLP techniques.
    
    Steps:
    1. Convert to lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove special characters and numbers
    4. Tokenize
    5. Remove stopwords
    6. Lemmatization
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except:
        return text

def predict_cyberbullying(text):
    """
    Predict if the given text contains cyberbullying.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Prediction result with label and confidence score
    """
    global model, vectorizer
    
    # If model not loaded, return demo result
    if model is None or vectorizer is None:
        # Demo mode - simple keyword-based detection
        bullying_keywords = ['hate', 'stupid', 'ugly', 'kill', 'die', 'loser', 
                           'idiot', 'dumb', 'worthless', 'pathetic', 'disgusting',
                           'threat', 'hurt', 'attack', 'bully']
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in bullying_keywords if keyword in text_lower)
        
        if keyword_count >= 2:
            return {
                'prediction': 'Cyberbullying Detected',
                'label': 'bullying',
                'confidence': min(0.85 + (keyword_count * 0.03), 0.98),
                'message': 'This text contains potentially harmful content.'
            }
        elif keyword_count == 1:
            return {
                'prediction': 'Potentially Harmful',
                'label': 'neutral',
                'confidence': 0.65,
                'message': 'This text may contain mildly harmful content.'
            }
        else:
            return {
                'prediction': 'Non-Bullying',
                'label': 'non-bullying',
                'confidence': 0.92,
                'message': 'This text appears to be safe and non-harmful.'
            }
    
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Transform using TF-IDF vectorizer
    text_vector = vectorizer.transform([cleaned_text])
    
    # Get prediction and probability
    prediction = model.predict(text_vector)[0]
    probabilities = model.predict_proba(text_vector)[0]
    confidence = max(probabilities)
    
    # Map prediction to labels
    labels = {0: 'Non-Bullying', 1: 'Cyberbullying Detected', 2: 'Potentially Harmful'}
    messages = {
        0: 'This text appears to be safe and non-harmful.',
        1: 'This text contains potentially harmful cyberbullying content.',
        2: 'This text may contain mildly harmful content.'
    }
    
    return {
        'prediction': labels.get(prediction, 'Unknown'),
        'label': 'bullying' if prediction == 1 else 'non-bullying' if prediction == 0 else 'neutral',
        'confidence': float(confidence),
        'message': messages.get(prediction, 'Unable to classify.')
    }


# ============ ROUTES ============

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/introduction')
def introduction():
    """Render the introduction page."""
    return render_template('introduction.html')

@app.route('/objectives')
def objectives():
    """Render the objectives page."""
    return render_template('objectives.html')

@app.route('/feasibility')
def feasibility():
    """Render the feasibility study page."""
    return render_template('feasibility.html')

@app.route('/srs')
def srs():
    """Render the Software Requirement Specification page."""
    return render_template('srs.html')

@app.route('/methodology')
def methodology():
    """Render the methodology page."""
    return render_template('methodology.html')

@app.route('/future_scope')
def future_scope():
    """Render the future scope page."""
    return render_template('future_scope.html')

@app.route('/detect')
def detect():
    """Render the detection page."""
    return render_template('detect.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for cyberbullying detection.
    
    Accepts POST request with JSON body containing 'text' field.
    Returns prediction result with label and confidence score.
    """
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'message': 'Please provide text to analyze.'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'error': 'Empty text',
                'message': 'Please enter some text to analyze.'
            }), 400
        
        # Get prediction
        result = predict_cyberbullying(text)
        result['original_text'] = text
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


# ============ MAIN ============

if __name__ == '__main__':
    # Load the ML model
    load_model()
    
    # Run the Flask application
    print("\n" + "="*50)
    print("Cyberbullying Detection Web Application")
    print("="*50)
    print("Starting server at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
