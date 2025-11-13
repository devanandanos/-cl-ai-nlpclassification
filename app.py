import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model + vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Enter a movie review and see if it’s Positive or Negative!")

review = st.text_area("✍️ Your review:")
if st.button("Analyze Sentiment"):
    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])
    sentiment = model.predict(X)[0]
    if sentiment == "positive":
        st.success("✅ Positive Review")
    else:
        st.error("❌ Negative Review")
