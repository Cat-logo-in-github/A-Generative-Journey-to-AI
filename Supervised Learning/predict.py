import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Parameters (must match training settings)
vocab_size = 10000
max_sequence_length = 200

# Load trained model
model = tf.keras.models.load_model("sentiment_model.keras")


def predict_sentiment(review_text):
    word_to_index = imdb.get_word_index()
    # IMDb indices are offset by 3 (0: padding, 1: start, 2: unknown)
    tokens = [word_to_index.get(word, -3) + 3 for word in review_text.lower().split()]
    tokens = [t if t < vocab_size else 2 for t in tokens] # Keep within vocab bounds
    
    padded_tokens = pad_sequences([tokens], maxlen=max_sequence_length)
    prediction = model.predict(padded_tokens)[0][0]
    
    return "Positive" if prediction > 0.5 else "Negative"


# Example usage
new_review = "This movie was fantastic! I really enjoyed it."
new_review = re.sub(r'[^\w\s]', '', new_review)  # Remove punctuation
sentiment = predict_sentiment(new_review)
print(f"Review: {new_review}")
print(f"Predicted Sentiment: {sentiment}")