import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Define paths to files
model_file_path = "best_model.pkl"
vectorizer_file_path = "vectorizer.pkl"

# Load the model and vectorizer
try:
    with open(model_file_path, "rb") as model_file:
        model = pickle.load(model_file)

    with open(vectorizer_file_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

except FileNotFoundError:
    st.error("Required files not found. Ensure 'best_model.pkl' and 'vectorizer.pkl' are in the correct directory.")
    st.stop()
except ValueError as ve:
    st.error(f"File content error: {ve}")
    st.stop()

# Function to preprocess and predict sentiment
def predict_sentiment(user_input):
    preprocessed_input = vectorizer.transform([user_input])  # Vectorize the input
    prediction = model.predict(preprocessed_input)  # Predict
    return "Positive" if prediction == 1 else "Negative"

# Streamlit interface
st.title("Sentiment Analysis App")
st.write("Enter a movie review below, and the app will predict whether the sentiment is Positive or Negative.")

# User input
user_input = st.text_area("Your Review:", "")

# Make predictions
if st.button("Analyze Sentiment"):
    if user_input.strip():
        try:
            result = predict_sentiment(user_input)
            st.subheader(f"Prediction: {result}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Please enter a valid review.")
