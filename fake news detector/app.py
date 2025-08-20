from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
import re
import string
from urllib.parse import urlparse, urljoin
import time
from datetime import datetime
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from textstat import flesch_reading_ease
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)

class OnlineFakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.model = LogisticRegression()
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.is_trained = False
        
        # Known reliable news sources
        self.reliable_sources = {
            'reuters.com', 'bbc.com', 'bbc.co.uk', 'apnews.com', 'npr.org',
            'cnn.com', 'nytimes.com', 'washingtonpost.com', 'theguardian.com',
            'wsj.com', 'bloomberg.com', 'abcnews.go.com', 'cbsnews.com',
            'nbcnews.com', 'usatoday.com', 'time.com', 'newsweek.com',
            'economist.com', 'pbs.org', 'axios.com'
        }
        
        # Suspicious patterns in fake news
        self.suspicious_patterns = [
            r'\bBREAKING\b.*!{2,}',
            r'\bSHOCKING\b.*!{2,}',
            r'\bEXCLUSIVE\b.*!{2,}',
            r'\bURGENT\b.*!{2,}',
            r'\bYOU WON\'T BELIEVE\b',
            r'\bDOCTORS HATE\b',
            r'\bTHIS ONE TRICK\b',
            r'\bTHEY DON\'T WANT YOU TO KNOW\b',
            r'\bCLICK HERE\b',
            r'\bSECRET REVEALED\b',
        ]
        
        self.train_model()
    
    def extract_content_from_url(self, url):
        """Extract article content from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Try to find the main article content
            article_selectors = [
                'article', '.article-content', '.story-body', '.entry-content',
                '.post-content', '.article-body', '.content', 'main',
                '.story-content', '.article-text'
            ]
            
            content = ""
            title = ""
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Try to find article content
            for selector in article_selectors:
                article_element = soup.select_one(selector)
                if article_element:
                    # Get all paragraph text
                    paragraphs = article_element.find_all(['p', 'h1', 'h2', 'h3'])
                    content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if len(content) > 200:  # Minimum content length
                        break
            
            # Fallback: get all paragraphs from the page
            if len(content) < 200:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'domain': urlparse(url).netloc.lower()
            }
            
        except Exception as e:
            return {'error': f"Failed to extract content: {str(e)}"}
    
    def analyze_source_credibility(self, domain):
        """Analyze source credibility based on domain"""
        domain = domain.lower().replace('www.', '')
        
        if any(reliable in domain for reliable in self.reliable_sources):
            return {'credible': True, 'score': 0.9, 'reason': 'Known reliable source'}
        
        # Check for suspicious domain patterns
        suspicious_indicators = [
            '.blogspot.', '.wordpress.', '.wix.', '.tumblr.',
            'fake', 'hoax', 'satire', 'conspiracy', 'truth',
            '24x7', 'breaking', 'viral', 'buzz'
        ]
        
        for indicator in suspicious_indicators:
            if indicator in domain:
                return {'credible': False, 'score': 0.2, 'reason': f'Suspicious domain pattern: {indicator}'}
        
        # Neutral assessment for unknown domains
        return {'credible': None, 'score': 0.5, 'reason': 'Unknown source'}
    
    def analyze_writing_style(self, text):
        """Analyze writing style for suspicious patterns"""
        if not text:
            return {'score': 0.5, 'indicators': []}
        
        indicators = []
        score = 0.5
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(f"Suspicious pattern: {pattern}")
                score -= 0.1
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.1:
            indicators.append(f"Excessive capitalization: {caps_ratio:.1%}")
            score -= 0.1
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '!?') / max(len(text), 1)
        if punct_ratio > 0.02:
            indicators.append(f"Excessive exclamation/question marks: {punct_ratio:.1%}")
            score -= 0.1
        
        # Check readability
        try:
            readability = flesch_reading_ease(text)
            if readability > 80:  # Very easy to read (possibly oversimplified)
                indicators.append(f"Unusually simple language (readability: {readability:.1f})")
                score -= 0.05
            elif readability < 30:  # Very difficult to read
                indicators.append(f"Unusually complex language (readability: {readability:.1f})")
                score -= 0.05
        except:
            pass
        
        return {'score': max(0, min(1, score)), 'indicators': indicators}
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        # Remove stopwords and stem
        words = text.split()
        words = [self.ps.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def train_model(self):
        """Train the model with sample data"""
        # Extended training data with more realistic examples
        fake_news = [
            "BREAKING: Scientists discover that vaccines contain microchips for government surveillance, study reveals shocking truth",
            "SHOCKING: Local man grows 50-foot tomatoes using this one weird trick doctors hate, you won't believe what happens next",
            "EXCLUSIVE: Celebrity caught in scandal that will shock you forever, insider sources confirm devastating details",
            "URGENT: Government hiding the truth about alien invasion happening next week, leaked documents expose conspiracy",
            "AMAZING: Cure for all diseases found in your kitchen cabinet, pharmaceutical companies don't want you to know",
            "BREAKING: President secretly replaced by robot double, sources confirm with photographic evidence",
            "SHOCKING: 5G towers causing coronavirus outbreak, study reveals devastating truth about technology",
            "EXCLUSIVE: Time traveler from 2030 warns about upcoming disasters, predictions will terrify you",
            "UNBELIEVABLE: This one simple trick will make you rich overnight, banks hate this secret method",
            "EXPOSED: Mainstream media hiding the real truth about climate change, conspiracy runs deeper than thought"
        ]
        
        real_news = [
            "Local city council approves new budget for infrastructure improvements in downtown area",
            "Stock market experiences moderate gains following quarterly economic report release",
            "University researchers publish peer-reviewed study on climate change effects in coastal regions",
            "New transportation policy aims to reduce traffic congestion during peak hours",
            "Healthcare workers receive recognition for their dedicated pandemic response efforts",
            "Technology company announces quarterly earnings results, showing steady growth trends",
            "Educational reforms proposed to improve student outcomes in public schools",
            "Weather service issues advisory for upcoming storm system affecting eastern regions",
            "International trade negotiations continue between major economic partners this week",
            "Scientific conference presents latest research findings on renewable energy technologies"
        ]
        
        # Create training dataset
        texts = fake_news + real_news
        labels = [1] * len(fake_news) + [0] * len(real_news)  # 1 = Fake, 0 = Real
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty texts
        valid_indices = [i for i, text in enumerate(processed_texts) if text.strip()]
        processed_texts = [processed_texts[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        if len(processed_texts) > 0:
            # Vectorize texts
            X = self.vectorizer.fit_transform(processed_texts)
            
            # Train model
            self.model.fit(X, labels)
            self.is_trained = True
        
        print("Model trained successfully!")
    
    def predict_text(self, text):
        """Predict if text is fake or real"""
        if not self.is_trained or not text.strip():
            return {"error": "Model not trained or empty text"}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if not processed_text.strip():
            return {"error": "Text too short or no meaningful content"}
        
        # Vectorize
        text_vector = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        
        fake_prob = probability[1] if len(probability) > 1 else 0.5
        real_prob = probability[0] if len(probability) > 1 else 0.5
        
        return {
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "confidence": {
                "fake": round(fake_prob * 100, 2),
                "real": round(real_prob * 100, 2)
            }
        }
    
    def analyze_url(self, url):
        """Comprehensive analysis of a news URL"""
        # Extract content
        content_data = self.extract_content_from_url(url)
        
        if 'error' in content_data:
            return content_data
        
        domain = content_data['domain']
        title = content_data['title']
        content = content_data['content']
        
        # Analyze source credibility
        source_analysis = self.analyze_source_credibility(domain)
        
        # Analyze writing style
        style_analysis = self.analyze_writing_style(title + " " + content)
        
        # Predict using ML model
        ml_prediction = self.predict_text(title + " " + content)
        
        if 'error' in ml_prediction:
            ml_prediction = {"prediction": "UNKNOWN", "confidence": {"fake": 50, "real": 50}}
        
        # Combine all analyses for final score
        final_score = 0.5
        
        if source_analysis['credible'] is True:
            final_score += 0.2
        elif source_analysis['credible'] is False:
            final_score -= 0.2
        
        final_score += (style_analysis['score'] - 0.5) * 0.3
        
        # Weight ML prediction
        ml_weight = ml_prediction['confidence']['real'] / 100
        final_score = final_score * 0.7 + ml_weight * 0.3
        
        final_score = max(0, min(1, final_score))
        
        return {
            'url': url,
            'domain': domain,
            'title': title,
            'content_length': len(content),
            'source_analysis': source_analysis,
            'style_analysis': style_analysis,
            'ml_prediction': ml_prediction,
            'final_assessment': {
                'credibility_score': round(final_score * 100, 1),
                'likely_fake': final_score < 0.4,
                'likely_real': final_score > 0.6,
                'uncertain': 0.4 <= final_score <= 0.6
            }
        }

# Initialize detector
detector = OnlineFakeNewsDetector()

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({"error": "Please provide a URL to analyze"})
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        result = detector.analyze_url(url)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Please provide text to analyze"})
        
        # Analyze writing style
        style_analysis = detector.analyze_writing_style(text)
        
        # Get ML prediction
        ml_prediction = detector.predict_text(text)
        
        return jsonify({
            'text_length': len(text),
            'style_analysis': style_analysis,
            'ml_prediction': ml_prediction,
            'analysis_type': 'text_only'
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/check_source', methods=['POST'])
def check_source():
    try:
        data = request.get_json()
        domain = data.get('domain', '').strip()
        
        if not domain:
            return jsonify({"error": "Please provide a domain to check"})
        
        analysis = detector.analyze_source_credibility(domain)
        return jsonify({
            'domain': domain,
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)