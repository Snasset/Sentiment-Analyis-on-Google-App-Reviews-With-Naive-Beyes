import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import nltk
from deep_translator import GoogleTranslator
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# --- Preprocessing Function ---
from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\s]', '', text)
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def casefoldingText(text):
    return text.lower()

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    return [word for word in tokens if word not in stop_words]

def stemmingText(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

def toSentence(tokens):
    return ' '.join(tokens)

# Fungsi final preprocess untuk dipakai
def preprocess(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    tokens = tokenizingText(text)
    tokens = filteringText(tokens)
    tokens = stemmingText(tokens)
    return toSentence(tokens)


# --- Load model & vectorizer ---
model = joblib.load("./model/naive_bayes_model_ros.pkl")
vectorizer = joblib.load("./model/vectorizer.pkl")

# --- Streamlit UI ---
st.title("📊 Sentiment Analysis App - Naive Bayes")

menu = st.sidebar.selectbox("Pilih Mode", ["Prediksi Kalimat", "Prediksi File CSV"])



if menu == "Prediksi Kalimat":
    st.subheader("🔍 Masukkan Kalimat")
    user_input = st.text_area("Tulis kalimat di sini...")
    language = st.radio("Pilih Bahasa Input", ["English", "Indonesian"])

    if st.button("Prediksi"):
        if user_input.strip() != "":
            if language == 'Indonesian':
                user_input = GoogleTranslator(source='auto', target = 'en').translate(user_input)
            processed = preprocess(user_input)
            vec = vectorizer.transform([processed])
            pred = model.predict(vec)[0]
            st.success(f"🎯 Hasil Prediksi: **{pred}**")
        else:
            st.warning("Teks kosong!")

elif menu == "Prediksi File CSV":
    st.subheader("📁 Upload File CSV")
    st.markdown("**Perhatian!** Pada baris pertama, usahakan itu adalah nama kolom seperti:")
    st.markdown("- Baris 1: `teks`")
    st.markdown("- Baris 2: `This app is good`")
    language = st.radio("Pilih Bahasa Input", ["English", "Indonesian"])
    uploaded_file = st.file_uploader("Upload CSV dengan kolom teks", type=['csv'])
    

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        text_col = st.selectbox("Pilih kolom teks:", df.columns)

        if st.button("Prediksi Semua"):
            df['Preprocessed'] = df[text_col].astype(str).apply(preprocess)
            vecs = vectorizer.transform(df['Preprocessed'])
            df['Prediction'] = model.predict(vecs)
            st.dataframe(df[[text_col, 'Prediction']])

            # Download hasil
            csv_result = df[[text_col, 'Prediction']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil sebagai CSV",
                data=csv_result,
                file_name='hasil_prediksi.csv',
                mime='text/csv'
            )