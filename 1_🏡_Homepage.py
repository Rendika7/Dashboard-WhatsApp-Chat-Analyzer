# Library Needed for the Project ------------------------------------------------------------
import streamlit as st # Streamlit for WebAppDevelopment

import pandas as pd
import numpy as np

import string
import regex
import emoji
from collections import Counter
from datetime import datetime
import calendar
import time

import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import warnings
warnings.filterwarnings('ignore')

# ================================================================================================================

# Mendefinisikan HTML untuk top bar
top_bar = """
<div style="background-color:#333; padding:5px;border-radius: 10px;margin:15px;">
    <h3 style="color:white; text-align:center;font-size: 35px">✨ AUTOMATIC WHATSAPP ANALYSIS 🔥</h3>
</div>
"""

# Menampilkan top bar sebagai komponen HTML
st.markdown(top_bar, unsafe_allow_html=True)


# ================================================================================================================


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


# ================================================================================================================


from PIL import Image
import base64
import io
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


def date_time(s):
    pattern = r'^([0-9]+)\/([0-9]+)\/([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -'
    result = regex.match(pattern, s)
    if result:
        return True
    return False

def find_author(s):
    s = s.split(":")
    if len(s) == 2:
        return True
    else:
        return False

def getDatapoint(line):
    splitline = line.split(' - ')
    dateTime = splitline[0]
    date, time = dateTime.split(", ")
    message = " ".join(splitline[1:])
    if find_author(message):
        splitmessage = message.split(": ")
        author = splitmessage[0]
        message = " ".join(splitmessage[1:])
    else:
        author = None
    return date, time, author, message


# ================================================================================================================  




# Cache the data loading process
@st.cache_data
def load_data(file):
    data = []
    # Mendapatkan path sementara file yang diunggah
    file_path = "temp_file.txt"
    with open(file_path, "wb") as temp_file:
        temp_file.write(file.getvalue())

    # Membaca file dan memproses datanya
    with open(file_path, encoding="utf-8") as fp:
        fp.readline()
        messageBuffer = []
        date, time, author = None, None, None
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            if date_time(line):
                if len(messageBuffer) > 0:
                    data.append([date, time, author, ' '.join(messageBuffer)])
                messageBuffer.clear()
                date, time, author, message = getDatapoint(line)
                messageBuffer.append(message)
            else:
                messageBuffer.append(line)
    df = pd.DataFrame(data, columns=["Date", 'Time', 'Author', 'Message'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Streamlit layout
st.title("Data File Reader")

# Upload file
uploaded_file = st.file_uploader("Upload your file here:", type=["txt"])

st.warning("Warning: upload the data first, and move the page to 'Whatsapp Analysis' to give you the result!")

if uploaded_file is None:
    pass
# Jika file sudah diunggah
elif uploaded_file is not None:
    # Memuat file menjadi DataFrame
    with st.spinner('Loading...'):
        # Load data and cache it
        df = load_data(uploaded_file)


# ================================================================================================================


# # Streamlit layout
# st.title("Data File Reader.")

# # Upload file
# uploaded_file = st.file_uploader("Upload your file here:", type=(["txt"]))

# st.warning("Warning: upload the data first, and move the page to 'Whatsapp Analysis' to give you the result!")



# if uploaded_file is None:
#     pass
# # Jika file sudah diunggah
# elif uploaded_file is not None:
#     # Memuat file menjadi DataFrame
#     data = []
#     with st.spinner('Loading...'):
#         # Mendapatkan path sementara file yang diunggah
#         file_path = "temp_file.txt"
#         with open(file_path, "wb") as temp_file:
#             temp_file.write(uploaded_file.getvalue())

#         # Membaca file dan memproses datanya
#         with open(file_path, encoding="utf-8") as fp:
#             fp.readline()
#             messageBuffer = []
#             date, time, author = None, None, None
#             while True:
#                 line = fp.readline()
#                 if not line:
#                     break
#                 line = line.strip()
#                 if date_time(line):
#                     if len(messageBuffer) > 0:
#                         data.append([date, time, author, ' '.join(messageBuffer)])
#                     messageBuffer.clear()
#                     date, time, author, message = getDatapoint(line)
#                     messageBuffer.append(message)
#                 else:
#                     messageBuffer.append(line)


# # ================================================================================================================  


    # df = pd.DataFrame(data, columns=["Date", 'Time', 'Author', 'Message'])
    # df['Date'] = pd.to_datetime(df['Date'])


# ================================================================================================================

    # Menambahkan CSS untuk mempercantik tampilan
    st.markdown(
        """
        <style>
            .dataframe {
                font-size: 18px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            td {
                max-width: 150px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Menentukan jumlah baris yang ditampilkan per halaman
    rows_per_page = st.slider("Rows per page:", 10, 100, 10)

    # Menentukan halaman DataFrame dengan tombol next/previous
    page_number = st.number_input("Page Number:", min_value=1, max_value=int(len(df) / rows_per_page) + 1, value=1, step=1)
    
        # Membagi aplikasi menjadi dua kolom
    col1, col2 = st.columns(2)

    # Kolom 1
    with col1:
        # Button "Previous"
        if st.button("Previous"):
            page_number = max(1, page_number - 1)

    # Kolom 2
    with col2:
        # Button "Next"
        if st.button("Next"):
            page_number = min(int(len(df) / rows_per_page) + 1, page_number + 1)

    # Menentukan indeks batas bawah dan atas untuk tampilan DataFrame
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    # Menampilkan DataFrame dengan paginasi
    st.table(df.iloc[start_idx:end_idx])


    # Initialization for global variabel which is can be used in every pages
    if 'Dataframe' not in st.session_state:
        st.session_state['Dataframe'] = df

    st.divider()

    # Streamlit layout
    st.title("Filter Messages by the Words you Want")

    # Input teks dari pengguna
    input_word = st.text_input("Enter a word to search in Message:")

    # Memeriksa dan menampilkan kalimat yang mengandung kata tersebut
    result_sentences = df[df['Message'].str.contains(input_word, case=False)]['Message'].tolist()

    if input_word is None:
        pass
    elif input_word and result_sentences:
        st.write(f"Sentences containing '{input_word}':")
        for sentence in result_sentences:
            st.write(sentence)
    elif input_word:
        st.write(f"No sentences found containing '{input_word}' in 'Message'.")


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
        <p><a href="https://www.linkedin.com/in/rendika-nurhartanto-s-882431218/" target="_blank">Visit my LinkedIn</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
