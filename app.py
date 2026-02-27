import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# ---- Load Model & Vectorizer ----
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# ---- Preprocessing Function ----
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

# ---- Streamlit UI ----
st.title("Tweet Sentiment Analyzer")

tweet = st.text_input("Enter a tweet:")

if st.button("Analyze"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet!")
    else:
        cleaned = clean(tweet)
        result = model.predict(vectorizer.transform([cleaned]))[0]

        if result == 4:
            st.success("😊 Positive")
        elif result == 0:
            st.error("😞 Negative")
        else:
            st.info("😐 Neutral")