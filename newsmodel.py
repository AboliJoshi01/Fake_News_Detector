from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Download NLTK resources (only once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.lower().strip().split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

# Flask setup
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "📰 Fake News Detection API is Live"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        text = input_data.get('text', '')
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        return jsonify({'result': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
