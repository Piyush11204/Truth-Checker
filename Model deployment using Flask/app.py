from flask import Flask, render_template, request, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField
from werkzeug.utils import secure_filename
import os
import datetime
from werkzeug.middleware.proxy_fix import ProxyFix

# Your ML model imports here (if any)
# import joblib
# model = joblib.load('your_model.pkl')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.wsgi_app = ProxyFix(app.wsgi_app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Form definition
class NewsForm(FlaskForm):
    txt = StringField('News Text')
    image = FileField('News Image', validators=[FileAllowed(['jpg', 'png'], 'Images only!')])
    document = FileField('News Document', validators=[FileAllowed(['pdf', 'docx'], 'PDFs and Word Docs only!')])
    url = StringField('News URL')

# Sample data for fake news indicators - in production, these would come from your model
fake_factors = [
    "Excessive use of emotional language",
    "Lack of verifiable sources",
    "Inconsistencies in narrative",
    "Clickbait headline structure"
]

# Sample data for real news references - in production, these would come from your model
references = [
    "Verified by multiple credible sources",
    "Contains proper citations and references",
    "Balanced presentation of facts",
    "Author credentials verified"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    form = NewsForm()
    
    # Initialize template variables with default values
    template_vars = {
        'form': form,
        'current_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Default values for safety
        'probabilities': {'fake': 45.0, 'real': 55.0},
        'confidence': 75.5,
        'analysis_data_points': 12,
        'fake_factors': fake_factors,
        'references': references,
        'sentiment_score': 48,
        'sentiment_label': 'Neutral',
        'emotional_score': 35,
        'emotional_label': 'Low',
        'sensationalism_score': 30,
        'sensationalism_label': 'Low',
        'complexity_score': 72,
        'complexity_label': 'Moderate',
        'bias_score': 35,
        'bias_label': 'Low',
        'source_transparency_score': 68,
        'source_transparency_label': 'Good',
        'recommendation_1': 'Cross-reference with established news sources or fact-checking websites.',
        'recommendation_2': 'Verify the credibility of the source or author.',
        'recommendation_3': 'Seek multiple perspectives for a balanced understanding.'
    }
    
    return render_template("index.html", **template_vars)

@app.route('/predict', methods=['POST'])
def predict():
    form = NewsForm()
    
    if form.validate_on_submit():
        txt = form.txt.data
        url = form.url.data
        image = form.image.data
        document = form.document.data
        
        image_url = None
        document_url = None
        document_name = None
        
        # Process uploaded image if any
        if image:
            filename = secure_filename(image.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)
            image_url = url_for('static', filename=f'uploads/{filename}')
        
        # Process uploaded document if any
        if document:
            filename = secure_filename(document.filename)
            document_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            document.save(document_path)
            document_url = url_for('static', filename=f'uploads/{filename}')
            document_name = filename
        
        # In a real application, you'd use your ML model to analyze the content here
        # For this example, we'll use dummy prediction (0 = fake, 1 = real)
        # result = model.predict(txt) # Replace with your actual model prediction
        
        # Dummy result for demonstration
        result = 1 if txt and len(txt) > 500 else 0
        
        # Calculate probabilities - in production, these would come from your model
        fake_prob = 35.0 if result == 1 else 75.0
        real_prob = 100 - fake_prob
        
        # Generate confidence score - in production, this would come from your model
        confidence = 82.3 if result == 1 else 68.9
        
        # Detailed analysis scores - in production, these would come from your model
        if result == 1:
            sentiment_score = 52
            sentiment_label = 'Neutral'
            emotional_score = 28
            emotional_label = 'Low'
            sensationalism_score = 20
            sensationalism_label = 'Very Low'
            complexity_score = 78
            complexity_label = 'High'
            bias_score = 25
            bias_label = 'Low'
            source_transparency_score = 82
            source_transparency_label = 'Excellent'
        else:
            sentiment_score = 68
            sentiment_label = 'Emotional'
            emotional_score = 72
            emotional_label = 'High'
            sensationalism_score = 65
            sensationalism_label = 'High'
            complexity_score = 45
            complexity_label = 'Low'
            bias_score = 70
            bias_label = 'High'
            source_transparency_score = 35
            source_transparency_label = 'Poor'
        
        # Recommendations - in production, these would be dynamically generated
        if result == 1:
            recommendation_1 = 'Continue to cross-reference with other sources for completeness.'
            recommendation_2 = 'Check for recent updates as this information may evolve.'
            recommendation_3 = 'Consider the context and potential biases even in reliable reporting.'
        else:
            recommendation_1 = 'Verify this information with established, credible news sources.'
            recommendation_2 = 'Check official websites or publications for verification.'
            recommendation_3 = 'Be cautious about sharing this content without verification.'
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis_data_points = 15
        
        return render_template('index.html', 
            form=form,
            result=result, 
            probabilities={'fake': fake_prob, 'real': real_prob}, 
            confidence=confidence,
            txt=txt, 
            url=url,
            image_url=image_url,
            document_url=document_url,
            document_name=document_name,
            timestamp=timestamp,
            fake_factors=fake_factors,
            references=references,
            analysis_data_points=analysis_data_points,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            emotional_score=emotional_score,
            emotional_label=emotional_label,
            sensationalism_score=sensationalism_score,
            sensationalism_label=sensationalism_label,
            complexity_score=complexity_score,
            complexity_label=complexity_label,
            bias_score=bias_score,
            bias_label=bias_label,
            source_transparency_score=source_transparency_score,
            source_transparency_label=source_transparency_label,
            recommendation_1=recommendation_1,
            recommendation_2=recommendation_2,
            recommendation_3=recommendation_3
        )
    
    # If form validation fails, return to the index page
    flash('Please check your input and try again.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)