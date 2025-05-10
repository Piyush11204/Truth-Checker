from flask import Flask, render_template, request, flash, url_for
import joblib
import re
import string
import pandas as pd
import os
import numpy as np
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import docx
import PyPDF2
import requests
from bs4 import BeautifulSoup
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'simple-secret-key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'static', 'Uploads')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload size

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def load_model():
    try:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "Model.pkl"),
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "Model.pkl"),
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "models", "Model.pkl")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                if hasattr(model, 'predict'):
                    dummy_text = pd.Series(["This is a test example"])
                    model.predict(dummy_text)
                    return model
        
        return create_fallback_model()
    
    except Exception:
        return create_fallback_model()

def create_fallback_model():
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    
    dummy_texts = [
        "Official government report confirms economic growth in Q3",
        "Scientific study reveals new climate change patterns",
        "Trusted news agency reports on diplomatic negotiations",
        "Breaking: Aliens invade Earth, claims anonymous source",
        "Shocking conspiracy: Government hides vaccine truth",
        "Click here to win millions in fake lottery scam",
        "Respected journal publishes peer-reviewed health study",
        "Unverified rumor suggests celebrity scandal",
        "You won't believe this miracle cure for all diseases",
        "International organization releases verified statistics"
    ]
    dummy_labels = [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]  # 1=real, 0=fake
    
    model.fit(dummy_texts, dummy_labels)
    return model

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return preprocess_text(text)
    except Exception:
        return ""

def extract_text_from_docx(docx_file):
    try:
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        return preprocess_text(text)
    except Exception:
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return preprocess_text(text)
    except Exception:
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
    except Exception:
        return ""

def analyze_content(text):
    fake_indicators = [
        ("unnamed sources", "Use of unnamed sources"),
        ("you won’t believe", "Clickbait phrasing"),
        ("shocking revelation", "Sensationalist language"),
        ("conspiracy", "Conspiracy theory references"),
        ("urgent warning", "Alarmist tone"),
        ("miracle cure", "Unsubstantiated claims")
    ]
    fake_factors = [desc for pattern, desc in fake_indicators if pattern in text.lower()]
    
    trusted_domains = ['bbc.com', 'reuters.com', 'nytimes.com', 'gov.', 'edu.', 'ndtv.com', 'thehindu.com', 'theguardian.com']
    references = [f"Source: {domain}" for domain in trusted_domains if domain in text.lower()]
    
    return fake_factors, references

def advanced_text_analysis(text):
    word_count = len(text.split())
    unique_words = len(set(text.split()))
    avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
    
    trusted_domains = ['bbc.com', 'reuters.com', 'nytimes.com', 'gov.', 'edu.', 'ndtv.com', 'thehindu.com', 'theguardian.com']
    
    sentiment_score = 50
    sentiment_label = "Neutral"
    positive_words = ["good", "great", "positive", "success", "verified"]
    negative_words = ["bad", "terrible", "negative", "fake", "hoax"]
    if any(word in text.lower() for word in positive_words):
        sentiment_score = 75
        sentiment_label = "Positive"
    elif any(word in text.lower() for word in negative_words):
        sentiment_score = 25
        sentiment_label = "Negative"
    
    sensationalism_score = 30
    sensationalism_label = "Low"
    sensational_words = ["shocking", "unbelievable", "amazing", "incredible"]
    if any(word in text.lower() for word in sensational_words):
        sensationalism_score = 80
        sensationalism_label = "High"
    
    complexity_score = min(100, int((unique_words / max(word_count, 1)) * 100))
    complexity_label = "Moderate"
    if complexity_score > 80:
        complexity_label = "High"
    elif complexity_score < 40:
        complexity_label = "Low"
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "sensationalism_score": sensationalism_score,
        "sensationalism_label": sensationalism_label,
        "complexity_score": complexity_score,
        "complexity_label": complexity_label,
        "readability_score": min(100, int(100 - (avg_word_length * 10))),
        "readability_label": "Good" if avg_word_length < 6 else "Complex",
        "credibility_score": 70 if any(domain in text.lower() for domain in trusted_domains) else 30,
        "credibility_label": "High" if any(domain in text.lower() for domain in trusted_domains) else "Low"
    }

MODEL = load_model()

@app.route('/')
def index():
    template_vars = {
        "result": None,
        "confidence": None,
        "probabilities": {"fake": 50.0, "real": 50.0},  # Default probabilities for initial load
        "txt": "",
        "url": "",
        "image_url": None,
        "document_url": None,
        "document_name": None,
        "fake_factors": [],
        "references": [],
        "timestamp": None,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_data_points": 10,
        **advanced_text_analysis(""),
        "recommendation_1": "Check primary sources for verification.",
        "recommendation_2": "Compare with reputable news outlets.",
        "recommendation_3": "Evaluate author credentials."
    }
    return render_template("index.html", **template_vars)

@app.route('/', methods=['POST'])
def predict():
    text = request.form.get('txt', '').strip()
    image = request.files.get('image')
    document = request.files.get('document')
    url = request.form.get('url', '').strip()
    
    if not (text or image or document or url):
        flash("Please provide consolidated least one input.", "warning")
        return render_template("index.html", txt=text, url=url, probabilities={"fake": 50.0, "real": 50.0})
    
    processed_text = ""
    image_url = document_url = document_name = None
    
    if text:
        processed_text = preprocess_text(text)
    
    if image and image.mimetype in ['image/jpeg', 'image/png']:
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        image_url = url_for('static', filename=f'Uploads/{filename}')
        processed_text += " " + extract_text_from_image(image_path)
    
    if document and document.mimetype in [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    ]:
        filename = secure_filename(document.filename)
        document_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        document.save(document_path)
        document_url = url_for('static', filename=f'Uploads/{filename}')
        document_name = filename
        extracted_text = (extract_text_from_pdf if document.mimetype == 'application/pdf' 
                        else extract_text_from_docx)(document_path)
        processed_text += " " + extracted_text
    
    if url:
        processed_text += " " + extract_text_from_url(url)
    
    if not processed_text.strip():
        flash("No meaningful text extracted.", "warning")
        return render_template("index.html", txt=text, url=url, probabilities={"fake": 50.0, "real": 50.0})
    
    fake_factors, references = analyze_content(processed_text)
    analysis_results = advanced_text_analysis(processed_text)
    text_series = pd.Series([processed_text])
    prediction = MODEL.predict(text_series)
    
    confidence = 50.0  # Default confidence
    probabilities = {"fake": 50.0, "real": 50.0}  # Default probabilities
    try:
        if hasattr(MODEL, 'predict_proba'):
            proba = MODEL.predict_proba(text_series)
            confidence = float(np.max(proba) * 100)
            probabilities = {
                'fake': float(proba[0][0] * 100),
                'real': float(proba[0][1] * 100)
            }
    except Exception:
        pass  # Use default values if predict_proba fails
    
    template_vars = {
        "result": int(prediction[0]),
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
        "analysis_data_points": 10,
        **analysis_results,
        "recommendation_1": "Check primary sources for verification.",
        "recommendation_2": "Compare with reputable news outlets.",
        "recommendation_3": "Evaluate author credentials."
    }
    
    return render_template("index.html", **template_vars)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)