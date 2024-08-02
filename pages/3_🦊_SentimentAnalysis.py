import streamlit as st # Streamlit for WebAppDevelopment
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import calendar
import time
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import regex
import emoji
from PIL import Image
import base64
import io

import warnings
warnings.filterwarnings('ignore')

# https://arnaudmiribel.github.io/streamlit-extras/

# Set Streamlit page configuration
st.set_page_config(layout="wide")


# Menampilkan gambar dengan posisi di tengah (center)
st.sidebar.markdown("""
    <style>
        div.stSidebar > div:first-child {
            display: flex;
            justify-content: center;
            text-align: center;
        }
        .sidebar-text {
            text-align: justify;
        }
        .h3-text {
            text-align: center;
        }
        .sidebar-button {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)



# Mendefinisikan HTML untuk top bar ------------------------------------------------------------
top_bar = """
<div style="background-color:#333; padding:5px;border-radius: 10px;">
    <h3 style="color:white; text-align:center;font-size: 35px"> 🤖 Sentiment Analysis for Whatsapp Chat Messages 🤖 </h3>
</div>
"""

# Menampilkan top bar sebagai komponen HTML
st.markdown(top_bar, unsafe_allow_html=True)


# Define the CSS for center alignment
st.markdown(
    """
    <style>
    .sidebar-text {
        text-align: center;
        font-size: 18px;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the text inside a container
st.markdown(
    "<div class='sidebar-text'> This page is used to determine the sentiment of each text sent by the user. Sentiment includes Negative Positive Neutral, but most likely not very accurate. Sentiment Analysis uses BERT Transformers namely mdhugol/indonesia-bert-sentiment-classification. </div>",
    unsafe_allow_html=True
)
st.divider()

# ================================================================================================================

# Mendapatkan gambar
image_path = 'Source/Profile.png'
image = Image.open(image_path)

# Mengonversi gambar menjadi format base64
img_buffer = io.BytesIO()
image.save(img_buffer, format="PNG")
img_str = base64.b64encode(img_buffer.getvalue()).decode()

# Menambahkan dekorasi dan menampilkan gambar di sidebar dengan align center
st.sidebar.markdown("""
    <div style="display: flex; justify-content: center;">
        <img src="data:image/png;base64,{}" alt="Profile Image" style="width:125px;">
    </div>
""".format(img_str), unsafe_allow_html=True)

# Menambahkan teks atau dekorasi tambahan di sidebar
st.sidebar.markdown("<h3 class='h3-text'> Rendika Nurhartanto Suharto </h3>", unsafe_allow_html=True)

# Menampilkan teks di sidebar dengan center alignment
st.sidebar.markdown("<p class='sidebar-text'> Welcome to my Personal Dashboard, a powerful tool to automatically analyze WhatsApp Group Chat dialogues! Just upload the .txt file of your WhatsApp conversations and enjoy deep understanding of your conversations. Let's start your smart analysis now! 🚀</p>", unsafe_allow_html=True)

# ================================================================================================================

# State untuk mengontrol tampilan konten di bawah tombol
content_visible = st.sidebar.button("More About Me 👨‍💻", key="more_about_me", help="Toggle Content Visibility")
# Fungsi untuk membuka link
def open_link(url):
    # Tampilkan konten di bawah tombol jika checkbox dicentang
    if content_visible:
        st.write(f"Link: [{url}]({url})")

if content_visible:
    # Menambahkan tombol interaktif di sidebar dengan ikon dan link
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("LinkedIn", key="linkedin", help="LinkedIn"):
            open_link("linkedin.com/in/rendika-nurhartanto-s-882431218/")

        if st.button("WhatsApp", key="whatsapp", help="WhatsApp"):
            open_link("https://wa.me/6281334814045")

    with col2:
        if st.button("Instagram", key="instagram", help="Instagram"):
            open_link("instagram.com/rendika__07/?hl=en")

        if st.button("GitHub", key="github", help="GitHub"):
            open_link("github.com/Rendika7")


# ================================================================================================================


if "Dataframe" not in st.session_state or st.session_state.Dataframe is None:
    pass
else:
    # Retrieve and process the DataFrame
    df = st.session_state['Dataframe']
    df = df.dropna()
    # ================================================================================================================

    # Convert Time column to datetime
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.time

    # Mengekstrak informasi Hari, Bulan, dan Tahun
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.strftime('%B')  # Menggunakan %B untuk mendapatkan nama bulan
    df['Year'] = df['Date'].dt.year
    # Menambahkan kolom 'Day_Name' yang berisi nama hari
    df['Day_Name'] = df['Date'].dt.day_name()
    # Changing the datatype of column "Day".
    df['Day_Name'] = df['Day_Name'].astype('category')

    # Memilih kolom yang diinginkan
    selected_columns = ['Date', 'Year', 'Month', 'Day', 'Day_Name', 'Time', 'Author', 'Message'] # Kolom yang ada di Backend nya
    df = df[selected_columns]
    
# ================================================================================================================

from transformers import pipeline

# Cache the pipeline loading
@st.cache_resource
def load_pipeline():
    return pipeline("text-classification", model="mdhugol/indonesia-bert-sentiment-classification")

# Load the pipeline
sentiment_pipeline = load_pipeline()

# Define label mapping
label_mapping = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}

# Streamlit layout
st.title("Indonesian Sentiment Analysis")

# CSS for center alignment
st.markdown(
    """
    <style>
    .result-container {
        text-align: center;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input text for prediction
text_input = st.text_area("Masukkan teks untuk analisis sentimen:", height=100)

# Predict sentiment function
def predict_sentiment(text):
    result = sentiment_pipeline(text)
    label = result[0]['label']
    score = result[0]['score']
    sentiment = label_mapping[label]
    return sentiment, score

# Creating columns for layout
col1, col2 = st.columns(2)

with col1:
    # Predict button
    if st.button("Predict Sentiment"):
        if text_input:
            sentiment, score = predict_sentiment(text_input)

            # Display the result in a container
            st.markdown(
                f"""
                <div class="result-container">
                    <h4>Hasil Analisis</h4>
                    <p><strong>Teks:</strong> {text_input}</p>
                    <p><strong>Sentimen:</strong> {sentiment} ({score * 100:.2f}%)</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("Silakan masukkan teks untuk dianalisis.")

with col2:
    # Redirect to the model page
    st.markdown(
        """
        <a href="https://huggingface.co/mdhugol/indonesia-bert-sentiment-classification" target="_blank">
            <button style="background-color: #007bff; color: white; padding: 7px 20px; border: none; border-radius: 4px; cursor: pointer;">
                Model Here...
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
    
    
st.divider()

# Streamlit layout
st.title("Sentiment Analysis for Dataframe")
st.warning('This is just for sample data, because if you run every data, the website will crash and it will take a long time for you to run all the data you have. My suggestion is, use about 15 to 30 rows for the experiment.', icon="⚠️")

# Input numerik dari pengguna
max_rows = df.shape[0]
row_number = st.number_input('Pilih nomor baris untuk prediksi:', min_value=1, max_value=max_rows, step=1)


# Apply sentiment analysis
@st.cache_data
def process_data(df):
    df_sample = df.sample(row_number)
    df_sample[['Sentiment', 'Score']] = df_sample['Message'].apply(lambda x: pd.Series(predict_sentiment(x)))
    return df_sample

if st.button('Predict Sentiment for Dataframe', key='predict_button'):
    df_sample = process_data(df)
    st.write(df_sample)
    sentiment_counts = Counter(df_sample['Sentiment'])
    # Membuat dataframe dari hasil count untuk keperluan visualisasi
    sentiment_df = pd.DataFrame.from_dict(sentiment_counts, orient='index').reset_index()
    sentiment_df.columns = ['Sentiment', 'Count']
    st.write(sentiment_df)
    # Menampilkan chart
    st.bar_chart(sentiment_df.set_index('Sentiment'))



# Define the CSS for the footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f9f9f9;
        color: #333;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the footer
st.markdown(
    """
    <div class="footer">
        <p>© 2024 Rendika Nurhartanto Suharto. All rights reserved.</p>
        <p><a href="https://www.linkedin.com/in/rendika-nurhartanto-s-882431218/" target="_blank">Visit our LinkedIn</a></p>
    </div>
    """,
    unsafe_allow_html=True
)