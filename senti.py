import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import logging


logging.basicConfig(level=logging.DEBUG)


@st.cache
def load_data():
    try:
        data = pd.read_csv('IMDB_Dataset.csv', encoding='latin-1')
        logging.debug(f'Loaded {data.shape[0]} rows')
        return data[['review', 'sentiment']]
    except FileNotFoundError as e:
        logging.error("File not found.")
        st.error("File not found.")
        return pd.DataFrame()  

data = load_data()

if not data.empty:
   
    data = data.sample(n=50000, random_state=42) 

    
    def preprocess_data(data):
        data['sentiment'] = data['sentiment'].map({'negative': 0, 'positive': 1})  
        logging.debug('Data preprocessing complete')
        return data

    data = preprocess_data(data)

   
    X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)
    logging.debug('Data split into train and test sets')

    vectorizer = CountVectorizer(stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    logging.debug('Data vectorization complete')

    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=10, random_state=42),
        'Logistic Regression': make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=500)) 
    }

    
    predictions = {}
    for name, clf in classifiers.items():
        clf.fit(X_train_vectorized, y_train)
        predictions[name] = clf.predict(X_test_vectorized)
        logging.debug(f'{name} classifier trained and predictions made')

   
    st.title('ANALYSIS ON MOVIE REVIEWS')

    st.write('## Dataset')
    st.write(data.head())

    st.write('## Predictions')
    for name, prediction in predictions.items():
        st.write(f'### {name}')
        st.write(f'Accuracy: {accuracy_score(y_test, prediction):.2f}')
        st.write(f'Classification Report:\n {classification_report(y_test, prediction)}')

    st.write('## Test Your Own Review')
    user_input = st.text_area('Enter a review:')
    if st.button('Predict'):
        if user_input:
            user_input_vectorized = vectorizer.transform([user_input])
            for name, clf in classifiers.items():
                st.write(f'### {name}')
                prediction = clf.predict(user_input_vectorized)
                sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
                st.write(f'Sentiment: {sentiment}')
