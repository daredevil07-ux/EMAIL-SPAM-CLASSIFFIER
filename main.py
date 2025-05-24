import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.markdown("""
    <style>
        
        body, .main, .stApp {
            background-color: #F4F4F4 !important;
        }

        
        .title {
            color: #000000 !important; /* Dark color for title */
            text-align: center;
            font-size: 32px;
            font-weight: bold;
        }

        
        h2, h3, h4, h5, h6, p, label {
            color: black;
        }

        
        .stTextArea textarea {
            background-color: #FFFFFF;
            color: black;
            border-radius: 10px;
            border: 1px solid #CCCCCC;
            padding: 10px;
            cursor: text !important;  /* Ensures text pointer */
            caret-color: black;  /* Makes sure caret (blinking cursor) is visible */
            outline: none;  /* Removes blue outline */
            pointer-events: auto !important; /* Ensures text area is interactive */
        }

        
        .stTextArea textarea:focus {
            border: 2px solid #4CAF50 !important; /* Adds green border when focused */
        }

        
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            transition: 0.3s;
            border: none;
        }

        .stButton button:hover {
            background-color: #45A049;
        }

        
        .stHeader {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='title'>📩 SMS Spam Classifier</h1>", unsafe_allow_html=True)


input_sms = st.text_area("Enter the message:")


if st.button('Predict'):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.markdown("<h2 class='stHeader' style='color: #E63946;'> Spam</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='stHeader' style='color: #4CAF50;'> Not Spam</h2>", unsafe_allow_html=True)
