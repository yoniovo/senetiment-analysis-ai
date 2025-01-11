from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask_cors import CORS


nltk.download('punkt')
nltk.download('stopwords')


app = Flask(__name__)


CORS(app, resources={r"/analyze": {"origins": "*"}})

model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    
    text = re.sub(r'<.*?>', '', text)
    
    
    text = re.sub(r'http\S+', '', text)
    
   
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
   
    text = text.lower()
    
    
    tokens = word_tokenize(text)
    
   
    tokens = [word for word in tokens if word not in stop_words]
    
  
    return " ".join(tokens)


def classify_with_threshold(review, model, vectorizer, threshold=0.2):
   
    review_cleaned = preprocess_text(review)
    
    
    review_tfidf = vectorizer.transform([review_cleaned])
    
    
    prob = model.predict_proba(review_tfidf)[0]
    
    positive_prob = prob[1]
    negative_prob = prob[0]
    
    
    if abs(positive_prob - negative_prob) < threshold:
        return 'neutral'
    elif positive_prob < negative_prob:
        return 'negative'
    else:
        return 'positive'

# API route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    try:
        # Get review from request JSON
        data = request.get_json()
        review = data.get('review', '')
        
        if not review:
            return jsonify({'error': 'No review provided'}), 400
        
        # Classify sentiment
        sentiment = classify_with_threshold(review, model, vectorizer)
        
        # Return the sentiment as JSON
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Root endpoint
@app.route('/')
def index():
    return jsonify({'message': 'Welcome to the Sentiment Analysis API! Use the /analyze endpoint to classify reviews.'})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
