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
from werkzeug.utils import secure_filename
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pytesseract
from PIL import Image
import docx
import PyPDF2
import requests
from bs4 import BeautifulSoup
import datetime

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
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'Uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size
app.wsgi_app = ProxyFix(app.wsgi_app)

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

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

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return preprocess_text(text)
    except Exception as e:
        app.logger.error(f"Error extracting text from image: {str(e)}")
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return preprocess_text(text)
    except Exception as e:
        app.logger.error(f"Error extracting text from docx: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return preprocess_text(text)
    except Exception as e:
        app.logger.error(f"Error extracting text from pdf: {str(e)}")
        return ""

def extract_text_from_url(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in ['header', 'nav', 'footer', 'script', 'style']:
            for element in soup.find_all(tag):
                element.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return preprocess_text(text)
    except Exception as e:
        app.logger.error(f"Error extracting text from URL: {str(e)}")
        return ""

def analyze_content(text):
    fake_indicators = [
        ("unnamed sources", "Use of unnamed sources"),
        ("you won’t believe", "Clickbait phrasing"),
        ("shocking revelation", "Sensationalist language"),
        ("conspiracy", "Conspiracy theory references"),
        ("urgent warning", "Alarmist tone")
    ]
    fake_factors = []
    for pattern, description in fake_indicators:
        if pattern in text.lower():
            fake_factors.append(description)
    
    references = []
    trusted_domains = ['bbc.com', 'reuters.com', 'nytimes.com', 'gov.', 'edu.']
    if any(domain in text.lower() for domain in trusted_domains):
        references = [f"Source: {domain}" for domain in trusted_domains if domain in text.lower()]
    
    return fake_factors, references

# Placeholder function for advanced text analysis (to be implemented)
def advanced_text_analysis(text):
    # TODO: Implement real analysis using NLP libraries like TextBlob, spaCy, or VADER
    # For now, return placeholder values based on simple heuristics
    sentiment_score = 48  # 0-100, neutral
    sentiment_label = "Neutral"
    if "positive" in text.lower() or "good" in text.lower():
        sentiment_score = 75
        sentiment_label = "Positive"
    elif "negative" in text.lower() or "bad" in text.lower():
        sentiment_score = 25
        sentiment_label = "Negative"
    
    emotional_score = 35  # 0-100
    emotional_label = "Low"
    if "shock" in text.lower() or "amazing" in text.lower():
        emotional_score = 70
        emotional_label = "High"
    
    sensationalism_score = 30  # 0-100
    sensationalism_label = "Low"
    if "you won't believe" in text.lower() or "shocking" in text.lower():
        sensationalism_score = 80
        sensationalism_label = "High"
    
    complexity_score = 72  # 0-100
    complexity_label = "Moderate"
    word_count = len(text.split())
    if word_count > 100:
        complexity_score = 85
        complexity_label = "High"
    elif word_count < 20:
        complexity_score = 40
        complexity_label = "Low"
    
    bias_score = 35  # 0-100
    bias_label = "Low"
    if "opinion" in text.lower() or "believe" in text.lower():
        bias_score = 60
        bias_label = "Moderate"
    
    source_transparency_score = 68  # 0-100
    source_transparency_label = "Good"
    if "source" in text.lower() or any(domain in text.lower() for domain in ['bbc.com', 'reuters.com']):
        source_transparency_score = 90
        source_transparency_label = "Excellent"
    elif "anonymous" in text.lower():
        source_transparency_score = 20
        source_transparency_label = "Poor"
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "emotional_score": emotional_score,
        "emotional_label": emotional_label,
        "sensationalism_score": sensationalism_score,
        "sensationalism_label": sensationalism_label,
        "complexity_score": complexity_score,
        "complexity_label": complexity_label,
        "bias_score": bias_score,
        "bias_label": bias_label,
        "source_transparency_score": source_transparency_score,
        "source_transparency_label": source_transparency_label
    }

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
    
    # Define default template variables
    template_vars = {
        "result": None,
        "confidence": None,
        "probabilities": {"fake": None, "real": None},
        "txt": "",
        "url": "",
        "image_url": None,
        "document_url": None,
        "document_name": None,
        "fake_factors": [],
        "references": [],
        "timestamp": None,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_data_points": 12,  # Placeholder
        "sentiment_score": None,
        "sentiment_label": None,
        "emotional_score": None,
        "emotional_label": None,
        "sensationalism_score": None,
        "sensationalism_label": None,
        "complexity_score": None,
        "complexity_label": None,
        "bias_score": None,
        "bias_label": None,
        "source_transparency_score": None,
        "source_transparency_label": None,
        "recommendation_1": "Cross-reference with established news sources or fact-checking websites.",
        "recommendation_2": "Verify the credibility of the source or author.",
        "recommendation_3": "Seek multiple perspectives for a balanced understanding."
    }
    
    return render_template("index.html", **template_vars)

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
        
        # Initialize variables
        text = request.form.get('txt', '').strip()
        image = request.files.get('image')
        document = request.files.get('document')
        url = request.form.get('url', '').strip()
        processed_text = ""
        image_url = None
        document_url = None
        document_name = None
        
        # Validate input
        if not (text or image or document or url):
            flash("Please provide at least one input (text, image, document, or URL).", "warning")
            return render_template("index.html", txt=text, url=url)
        
        # Process inputs
        try:
            if text:
                processed_text = preprocess_text(text)
                app.logger.info(f"Processed text input: {processed_text[:50]}...")
            
            if image:
                if image.mimetype not in ['image/jpeg', 'image/png']:
                    flash("Invalid image format. Please upload a jpg or png file.", "warning")
                    return render_template("index.html", txt=text, url=url)
                filename = secure_filename(image.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(image_path)
                image_url = url_for('static', filename=f'Uploads/{filename}')
                extracted_text = extract_text_from_image(image_path)
                processed_text += " " + extracted_text if processed_text else extracted_text
                app.logger.info(f"Processed image input: {extracted_text[:50]}...")
            
            if document:
                if document.mimetype not in ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    flash("Invalid document format. Please upload a docx or pdf file.", "warning")
                    return render_template("index.html", txt=text, url=url)
                filename = secure_filename(document.filename)
                document_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                document.save(document_path)
                document_url = url_for('static', filename=f'Uploads/{filename}')
                document_name = filename
                if document.mimetype == 'application/pdf':
                    extracted_text = extract_text_from_pdf(document_path)
                else:
                    extracted_text = extract_text_from_docx(document_path)
                processed_text += " " + extracted_text if processed_text else extracted_text
                app.logger.info(f"Processed document input: {extracted_text[:50]}...")
            
            if url:
                extracted_text = extract_text_from_url(url)
                processed_text += " " + extracted_text if processed_text else extracted_text
                app.logger.info(f"Processed URL input: {extracted_text[:50]}...")
            
            # Validate processed text
            if not processed_text or processed_text.strip() == '':
                flash("No meaningful text could be extracted from the provided inputs.", "warning")
                return render_template("index.html", txt=text, url=url)
            
            # Analyze content for fake factors and references
            fake_factors, references = analyze_content(processed_text)
            
            # Perform advanced text analysis
            analysis_results = advanced_text_analysis(processed_text)
            
            # Perform prediction
            text_series = pd.Series([processed_text])
            prediction = MODEL.predict(text_series)
            
            # Get confidence score and prediction probabilities
            confidence = None
            probabilities = {"fake": None, "real": None}
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
            app.logger.info(f"Prediction result: {result}, Confidence: {confidence}, Probabilities: {probabilities}")
            
            # Prepare template variables
            template_vars = {
                "result": result,
                "confidence": confidence,
                "probabilities": probabilities,
                "txt": text,
                "url": url,
                "image_url": image_url,
                "document_url": document_url,
                "document_name": document_name,
                "fake_factors": fake_factors or ["No specific factors identified"],
                "references": references or ["No verified sources identified"],
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_data_points": 12,  # Placeholder
                "sentiment_score": analysis_results["sentiment_score"],
                "sentiment_label": analysis_results["sentiment_label"],
                "emotional_score": analysis_results["emotional_score"],
                "emotional_label": analysis_results["emotional_label"],
                "sensationalism_score": analysis_results["sensationalism_score"],
                "sensationalism_label": analysis_results["sensationalism_label"],
                "complexity_score": analysis_results["complexity_score"],
                "complexity_label": analysis_results["complexity_label"],
                "bias_score": analysis_results["bias_score"],
                "bias_label": analysis_results["bias_label"],
                "source_transparency_score": analysis_results["source_transparency_score"],
                "source_transparency_label": analysis_results["source_transparency_label"],
                "recommendation_1": "Cross-reference with established news sources or fact-checking websites.",
                "recommendation_2": "Verify the credibility of the source or author.",
                "recommendation_3": "Seek multiple perspectives for a balanced understanding."
            }
            
            return render_template("index.html", **template_vars)
            
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            flash(f"An error occurred during analysis. Please try again.", "error")
            return render_template("index.html", txt=text, url=url)
        
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