import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stop_words = set(stopwords.words('english'))  # Load English stop words

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove all non-alphabetic characters
    tokens = text.split()  # Split the text into tokens
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return " ".join(tokens)  # Join the tokens back into a single string

st.set_page_config(page_title="Sentiment analysis for movies", layout="centered")
st.title("Sentiment Analysis for Movies")
st.subheader("Enter a movie review to analyze its sentiment")

user_input = st.text_area("Enter the review :")

if st.button("Analyze"):
    cleaned_input = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    st.success(f"The sentiment of the review is: {sentiment}")

