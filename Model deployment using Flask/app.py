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
from urllib.parse import urlparse

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
        return preprocess_text(text), soup
    except Exception:
        return "", None

def search_references(text, url=None):
    """Search for corroborating references on trusted news sites."""
    trusted_domains = [
        'bbc.com', 'reuters.com', 'nytimes.com', 'theguardian.com',
        'ndtv.com', 'thehindu.com', 'apnews.com', 'npr.org', 'cnn.com',
        'aljazeera.com'
    ]
    references = []
    search_terms = ' '.join(text.split()[:5])  # Use first 5 words as search query
    search_url = f"https://www.google.com/search?q={search_terms}+site:({'|'.join(trusted_domains)})"
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if any(domain in href for domain in trusted_domains):
                # Extract URL from Google's redirect
                if href.startswith('/url?q='):
                    href = href.split('/url?q=')[1].split('&')[0]
                references.append(f"Source: {href}")
                if len(references) >= 3:  # Limit to 3 references
                    break
    except Exception:
        pass
    
    # Add references from the provided URL's domain if it's trusted
    if url:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        for trusted in trusted_domains:
            if trusted in domain:
                references.append(f"Source: {url}")
                break
    
    return references if references else ["No verified sources found"]

def analyze_content(text, url=None, soup=None):
    """Analyze content for fake indicators and references."""
    fake_indicators = [
        ("unnamed sources", "Use of unnamed sources"),
        ("you won’t believe", "Clickbait phrasing"),
        ("shocking revelation", "Sensationalist language"),
        ("conspiracy", "Conspiracy theory references"),
        ("urgent warning", "Alarmist tone"),
        ("miracle cure", "Unsubstantiated claims"),
        ("anonymous", "Lack of author attribution"),
        ("exclusive", "Exaggerated exclusivity"),
        ("breaking news", "Overuse of urgent terms")
    ]
    fake_factors = [desc for pattern, desc in fake_indicators if pattern in text.lower()]
    
    # Check for metadata if URL and soup are provided
    metadata_factors = []
    if soup:
        # Check for author
        author = soup.find('meta', {'name': 'author'}) or soup.find('meta', {'property': 'article:author'})
        if not author or not author.get('content', '').strip():
            metadata_factors.append("Missing author information")
        
        # Check for publication date
        pub_date = soup.find('meta', {'property': 'article:published_time'}) or soup.find('meta', {'name': 'date'})
        if not pub_date or not pub_date.get('content', '').strip():
            metadata_factors.append("Missing publication date")
    
    fake_factors.extend(metadata_factors)
    
    references = search_references(text, url)
    
    return fake_factors, references

def advanced_text_analysis(text, references):
    """Perform advanced text analysis for credibility metrics."""
    word_count = len(text.split())
    unique_words = len(set(text.split()))
    avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
    
    trusted_domains = [
        'bbc.com', 'reuters.com', 'nytimes.com', 'gov.', 'edu.', 'ndtv.com',
        'thehindu.com', 'theguardian.com', 'apnews.com', 'npr.org', 'cnn.com',
        'aljazeera.com'
    ]
    
    # Sentiment analysis
    sentiment_score = 50
    sentiment_label = "Neutral"
    positive_words = ["good", "great", "positive", "success", "verified", "trusted"]
    negative_words = ["bad", "terrible", "negative", "fake", "hoax", "scam"]
    if any(word in text.lower() for word in positive_words):
        sentiment_score = 75
        sentiment_label = "Positive"
    elif any(word in text.lower() for word in negative_words):
        sentiment_score = 25
        sentiment_label = "Negative"
    
    # Sensationalism
    sensationalism_score = 30
    sensationalism_label = "Low"
    sensational_words = ["shocking", "unbelievable", "amazing", "incredible", "outrageous"]
    if any(word in text.lower() for word in sensational_words):
        sensationalism_score = 80
        sensationalism_label = "High"
    
    # Complexity
    complexity_score = min(100, int((unique_words / max(word_count, 1)) * 100))
    complexity_label = "Moderate"
    if complexity_score > 80:
        complexity_label = "High"
    elif complexity_score < 40:
        complexity_label = "Low"
    
    # Readability
    readability_score = min(100, int(100 - (avg_word_length * 10)))
    readability_label = "Good" if avg_word_length < 6 else "Complex"
    
    # Credibility
    credibility_score = 70 if any(domain in text.lower() for domain in trusted_domains) or references != ["No verified sources found"] else 30
    credibility_label = "High" if credibility_score >= 70 else "Low"
    
    # Factual consistency (basic check for entities)
    factual_score = 50
    factual_label = "Moderate"
    entity_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # Dates (YYYY-MM-DD)
        r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names
        r'\b\d{1,3}(?:,\d{3})*\b'  # Numbers with commas
    ]
    if any(re.search(pattern, text) for pattern in entity_patterns):
        factual_score = 80
        factual_label = "High"
    
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "sensationalism_score": sensationalism_score,
        "sensationalism_label": sensationalism_label,
        "complexity_score": complexity_score,
        "complexity_label": complexity_label,
        "readability_score": readability_score,
        "readability_label": readability_label,
        "credibility_score": credibility_score,
        "credibility_label": credibility_label,
        "factual_score": factual_score,
        "factual_label": factual_label
    }

def get_decision_basis(prediction, probabilities, fake_factors, references, analysis_results):
    """Generate a list explaining the basis for the classification."""
    basis = []
    
    if prediction == 1:
        basis.append("Classified as real news due to the following factors:")
        if probabilities['real'] > 60:
            basis.append(f"- High model confidence in real news ({probabilities['real']:.1f}%).")
        if references != ["No verified sources found"]:
            basis.append("- Corroborating references found from trusted sources.")
        if analysis_results['credibility_score'] >= 70:
            basis.append("- Content aligns with credible sources or domains.")
        if analysis_results['factual_score'] >= 70:
            basis.append("- Presence of verifiable entities (e.g., dates, names).")
        if analysis_results['sensationalism_score'] < 50:
            basis.append("- Low sensationalist language detected.")
    else:
        basis.append("Classified as fake news due to the following factors:")
        if probabilities['fake'] > 60:
            basis.append(f"- High model confidence in fake news ({probabilities['fake']:.1f}%).")
        if fake_factors:
            basis.append(f"- Detected misleading indicators: {', '.join(fake_factors)}.")
        if references == ["No verified sources found"]:
            basis.append("- No corroborating references from trusted sources.")
        if analysis_results['sensationalism_score'] >= 70:
            basis.append("- High sensationalist language detected.")
        if analysis_results['credibility_score'] < 50:
            basis.append("- Lack of alignment with credible sources.")
    
    return basis if basis else ["No specific decision basis available."]

MODEL = load_model()

@app.route('/')
def index():
    template_vars = {
        "result": None,
        "confidence": None,
        "probabilities": {"fake": 50.0, "real": 50.0},
        "txt": "",
        "url": "",
        "image_url": None,
        "document_url": None,
        "document_name": None,
        "fake_factors": [],
        "references": [],
        "decision_basis": [],
        "timestamp": None,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_data_points": 10,
        **advanced_text_analysis("", []),
        "recommendation_1": "Cross-reference with primary sources.",
        "recommendation_2": "Verify with reputable news outlets.",
        "recommendation_3": "Check author and publication credibility."
    }
    return render_template("index.html", **template_vars)

@app.route('/', methods=['POST'])
def predict():
    text = request.form.get('txt', '').strip()
    image = request.files.get('image')
    document = request.files.get('document')
    url = request.form.get('url', '').strip()
    
    if not (text or image or document or url):
        flash("Please provide at least one input.", "warning")
        return render_template("index.html", txt=text, url=url, probabilities={"fake": 50.0, "real": 50.0}, decision_basis=[])
    
    processed_text = ""
    soup = None
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
        extracted_text, soup = extract_text_from_url(url)
        processed_text += " " + extracted_text
    
    if not processed_text.strip():
        flash("No meaningful text extracted.", "warning")
        return render_template("index.html", txt=text, url=url, probabilities={"fake": 50.0, "real": 50.0}, decision_basis=[])
    
    fake_factors, references = analyze_content(processed_text, url, soup)
    analysis_results = advanced_text_analysis(processed_text, references)
    text_series = pd.Series([processed_text])
    prediction = MODEL.predict(text_series)
    
    confidence = 50.0
    probabilities = {"fake": 50.0, "real": 50.0}
    try:
        if hasattr(MODEL, 'predict_proba'):
            proba = MODEL.predict_proba(text_series)
            confidence = float(np.max(proba) * 100)
            probabilities = {
                'fake': float(proba[0][0] * 100),
                'real': float(proba[0][1] * 100)
            }
    except Exception:
        pass
    
    decision_basis = get_decision_basis(prediction[0], probabilities, fake_factors, references, analysis_results)
    
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
        "references": references,
        "decision_basis": decision_basis,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_data_points": 12,  # Increased due to new metrics
        **analysis_results,
        "recommendation_1": "Cross-reference with primary sources.",
        "recommendation_2": "Verify with reputable news outlets.",
        "recommendation_3": "Check author and publication credibility."
    }
    
    return render_template("index.html", **template_vars)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)