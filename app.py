from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 


# Define the sentiment prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
    # Get the review from the request
        data = request.get_json(force=True)
        review = data.get('review','')
        if not review:
            return jsonify({"error":"No review provided"}),400
        
        # Preprocess the review (TF-IDF vectorization)
        review_vector = vectorizer.transform([review]).toarray()
        
        # Predict the sentiment using the SVM model
        prediction = svm_model.predict(review_vector)
        
        # Convert numeric sentiment to text label
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_sentiment = sentiment_map[prediction[0]]
        
        # Return the result as JSON
        return jsonify({"review": review, "sentiment": predicted_sentiment})
    except Exception as e:
        return jsonify({"error":str(e)}),500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
