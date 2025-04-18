from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import re
import string
import pandas as pd
import os
import logging
from logging.handlers import RotatingFileHandler
import numpy as np
from werkzeug.middleware.proxy_fix import ProxyFix
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configure logging
def setup_logging():
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/fake_news_detector.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Fake News Detector startup')

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'you-should-change-this-in-production'
app.wsgi_app = ProxyFix(app.wsgi_app)

if not app.debug:
    setup_logging()

def load_model():
    try:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "Model.pkl"),
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "Model.pkl"),
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "models", "Model.pkl")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                app.logger.info(f"Loading model from: {path}")
                try:
                    model = joblib.load(path)
                    if hasattr(model, 'predict'):
                        try:
                            dummy_text = pd.Series(["This is a test example"])
                            model.predict(dummy_text)
                            app.logger.info("Model successfully validated")
                            return model
                        except Exception as e:
                            app.logger.error(f"Model loaded but failed validation: {str(e)}")
                    else:
                        app.logger.error("Loaded object is not a valid model")
                except Exception as e:
                    app.logger.error(f"Error loading model from {path}: {str(e)}")
        
        app.logger.warning("Creating fallback model as no valid model was found")
        return create_fallback_model()
    
    except Exception as e:
        app.logger.error(f"Failed to load the model: {str(e)}")
        return create_fallback_model()

def create_fallback_model():
    app.logger.info("Creating a basic fallback model")
    fallback_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ])
    
    dummy_texts = [
        "This is real news about politics and events",
        "Breaking news from reliable sources about economy",
        "Factual reporting on international relations",
        "Fake news conspiracy theories aliens control government",
        "Click bait fake headlines shocking revelations",
        "You won't believe this outrageous fake claim"
    ]
    dummy_labels = [1, 1, 1, 0, 0, 0]
    
    fallback_model.fit(dummy_texts, dummy_labels)
    app.logger.info("Fallback model created and fitted")
    return fallback_model
# In your model training code:
def enhance_model():
    # Add more fake news indicators to training data
    fake_patterns = [
        ("unnamed sources", 0.9),  # High weight for unnamed sources
        ("according to leaked documents", 0.85),
        ("resistance is futile", 1.0),  # Very high weight for known fake phrases
        ("mandatory implantation", 0.8),
        ("jail time for non-compliance", 0.75)
    ]
    
    # Add more reliable sources to training data
    real_news_samples = scrape_trusted_news_sources()
    
    # Add sentiment analysis component
    from textblob import TextBlob
    def sentiment_analysis(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    try:
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub("\\W", " ", text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        app.logger.error(f"Error in text preprocessing: {str(e)}")
        return text

MODEL = None

@app.route('/')
def index():
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            app.logger.error(f"Failed to load model: {str(e)}")
            flash("The prediction model is currently unavailable. Please try again later.", "error")
    
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        global MODEL
        if MODEL is None:
            try:
                MODEL = load_model()
            except Exception as e:
                app.logger.error(f"Failed to load model on demand: {str(e)}")
                flash("The prediction model is currently unavailable. Please try again later.", "error")
                return render_template("index.html")
        
        raw_text = request.form.get('txt', '')
        
        if not raw_text or raw_text.strip() == '':
            flash("Please enter some text to analyze.", "warning")
            return render_template("index.html")
            
        safe_log_text = raw_text[:50] + '...' if len(raw_text) > 50 else raw_text
        app.logger.info(f"Processing new prediction request: {safe_log_text}")
        
        try:
            processed_text = preprocess_text(raw_text)
            
            if not processed_text or processed_text.strip() == '':
                flash("After preprocessing, no meaningful text remained. Please try with different content.", "warning")
                return render_template("index.html", input_text=raw_text)
            
            text_series = pd.Series([processed_text])
            prediction = MODEL.predict(text_series)
            
            # Get confidence score and prediction probabilities
            confidence = None
            probabilities = None
            
            if hasattr(MODEL, 'predict_proba'):
                try:
                    proba = MODEL.predict_proba(text_series)
                    confidence = float(np.max(proba) * 100)
                    probabilities = {
                        'fake': float(proba[0][0] * 100),
                        'real': float(proba[0][1] * 100)
                    }
                except Exception as e:
                    app.logger.warning(f"Couldn't get prediction probability: {str(e)}")
            
            result = int(prediction[0])
            app.logger.info(f"Prediction result: {result}, Confidence: {confidence}")
            
            return render_template(
                "index.html", 
                result=result,
                confidence=confidence,
                probabilities=probabilities,
                input_text=raw_text
            )
            
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            flash(f"An error occurred during analysis. Please try again.", "error")
            return render_template("index.html", input_text=raw_text)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)