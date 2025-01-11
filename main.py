import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

words_index = imdb.get_word_index()
reversed_word_index = {value: key for key,value in words_index.items()}

model = load_model('simple_rnn_imdb.h5')



st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area('Movie Review')


    

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [words_index.get(word, 2) +3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_text = preprocess_text(review)
    prediction = model.predict(preprocessed_text)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]


if st.button('Classify'):
    preprocess = preprocess_text(user_input)

    prediction = model.predict(preprocess)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

    st.write(f'Sentiment: {sentiment}')

