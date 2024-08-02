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

import warnings
warnings.filterwarnings('ignore')

# https://arnaudmiribel.github.io/streamlit-extras/

# Set Streamlit page configuration
st.set_page_config(layout="wide")
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


# Mendefinisikan HTML untuk top bar ------------------------------------------------------------
top_bar = """
<div style="background-color:#333; padding:5px;border-radius: 10px;">
    <h3 style="color:white; text-align:center;font-size: 35px"> The Analysis of Whatsapp Group Chat🔥</h3>
</div>
"""

# Menampilkan top bar sebagai komponen HTML
st.markdown(top_bar, unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.header("Filters")

# ================================================================================================================

if "Dataframe" not in st.session_state or st.session_state.Dataframe is None:
    # Add space using HTML
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("Warning: Upload your data on the home page first, after that you can return here 😁")
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

    # columns_to_view = ['Date', 'Time', 'Author', 'Message'] # kolom yang tampil ke UI Website
    # ================================================================================================================

    # Sidebar columns for time filter
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_time = st.time_input("Start Time:", df['Time'].min())
    with col2:
        end_time = st.time_input("End Time:", df['Time'].max())

    # ================================================================================================================

    # Sidebar for author filter
    authors = df['Author'].unique().tolist()
    authors.insert(0, 'All')  # Add an option for displaying all authors
    selected_author = st.sidebar.selectbox("Filter by Author:", authors)

    # Sidebar for date filter
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    selected_date = st.sidebar.date_input("Filter by Date:", [min_date, max_date], min_value=min_date, max_value=max_date)

    # Sidebar for Day_Name filter
    day_names = df['Day_Name'].unique().tolist()
    day_names.insert(0, 'All')
    selected_day_name = st.sidebar.selectbox("Filter by Day Name:", day_names)

    # Sidebar for Month filter
    months = df['Month'].unique().tolist()
    months.insert(0, 'All')
    selected_month = st.sidebar.selectbox("Filter by Month:", months)

    # Sidebar for Year filter
    years = df['Year'].unique().tolist()
    years.insert(0, 'All')
    selected_year = st.sidebar.selectbox("Filter by Year:", years)

    # ================================================================================================================

    # Apply the author filter
    if selected_author != 'All':
        df = df[df['Author'] == selected_author]

    # Apply the date filter
    start_date, end_date = selected_date
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

    # Apply the time filter
    df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]

    # Apply the Day_Name filter
    if selected_day_name != 'All':
        df = df[df['Day_Name'] == selected_day_name]

    # Apply the Month filter
    if selected_month != 'All':
        df = df[df['Month'] == selected_month]

    # Apply the Year filter
    if selected_year != 'All':
        df = df[df['Year'] == selected_year]

    # ================================================================================================================

    # # Create a new DataFrame with only the columns to view
    # df_view = df[columns_to_view]

    # ================================================================================================================

    # Add space using HTML
    st.markdown("<br>", unsafe_allow_html=True)

    # Display the DataFrame
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
    rows_per_page = st.slider("Rows per page:", 10, 100, 20)

    # Menentukan halaman DataFrame dengan tombol next/previous
    page_number = st.number_input("Page Number:", min_value=1, max_value=int(len(df) / rows_per_page) + 1, value=1, step=1)

    # Menentukan indeks batas bawah dan atas untuk tampilan DataFrame
    start_idx = (page_number - 1) * rows_per_page
    end_idx = start_idx + rows_per_page

    # # Menampilkan DataFrame dengan paginasi
    # st.table(df.iloc[start_idx:end_idx])

    # ================================================================================================================

    # Letter, Word, Url_count, Media_count, And Emoji ------ The added columns include **"Letter's"** (message length), **"Word's"**    (number of words in the message), **"URL_Count"** (number of links in the message), and **"Media_Count"** (number of media in the  message).

    ### Menghitung jumlah huruf dalam setiap pesan
    df["Letter's"] = df['Message'].apply(lambda s : len(s))
    ### Menghitung jumlah kata dalam setiap pesan
    df["Word's"] = df['Message'].apply(lambda s : len(s.split(' ')))

    ### Menghitung jumlah tautan dalam dataset pada setiap pesan
    URLPATTERN = r'(https?://\S+)'
    df['Url_Count'] = df.Message.apply(lambda x: regex.findall(URLPATTERN, x)).str.len()
    links = np.sum(df.Url_Count)

    ### Menghitung jumlah Emoji dalam dataset pada setiap pesan
    def split_count(text):
        emoji_list = []
        data = regex.findall(r'\X', text)
        for word in data:
            if emoji.is_emoji(word):
                emoji_list.append(word)
        return emoji_list

    df["Emoji's"] = df["Message"].apply(split_count)
    emojis = sum(df["Emoji's"].apply(len))

    ### Fungsi untuk menghitung jumlah media dalam obrolan.
    MEDIAPATTERN = r'<Media omitted>'
    df['Media_Count'] = df.Message.apply(lambda x : regex.findall(MEDIAPATTERN, x)).str.len()
    media = np.sum(df.Media_Count)

    # Menampilkan DataFrame dengan paginasi
    st.table(df.iloc[start_idx:end_idx])

    # ================================================================================================================
    ## Exploratory Data Analysis (EDA)

    ### Insight 1 - Simple Explore for Dataset ------------------------------------------------------------

    # HTML to center the subheader
    st.markdown("<h2 style='text-align: center;'>Simple Explore for Dataset</h2>", unsafe_allow_html=True)

    # Calculating metrics
    num_columns = df.shape[1]
    num_rows = df.shape[0]
    total_messages = df['Message'].count()
    total_authors = df['Author'].nunique()
    total_letters = df['Letter\'s'].sum()
    total_words = df['Word\'s'].sum()
    total_urls = df['Url_Count'].sum()
    total_emojis = sum(len(emojis) for emojis in df['Emoji\'s'])
    total_media = df['Media_Count'].sum()

    # Displaying metrics using Streamlit
    col1, col2, col3 = st.columns(3)

    col1.metric("1] Number of Columns", num_columns)
    col1.metric("2] Total Messages", total_messages)
    col1.metric("3] Total Letters", total_letters)

    col2.metric("4] Number of Rows", num_rows)
    col2.metric("5] Total Authors", total_authors)
    col2.metric("6] Total Words", total_words)

    col3.metric("7] Total URLs", total_urls)
    col3.metric("8] Total Emojis", total_emojis)
    col3.metric("9] Total Media", total_media)


    ### Insight 2 - Messages sent per day over a time period ------------------------------------------------------------

    df1 = df.copy()  # I will be using a copy of the original data frame everytime, to avoid loss of data!
    df1['message_count'] = 1  # adding extra helper column --> message_count.

    # Grouping by Date
    df1 = df1[['Date', 'message_count']].groupby('Date').sum().reset_index()

    # Adding Day_Name, Month, and Year columns
    df1['Day_Name'] = df1['Date'].dt.day_name()
    # Changing the datatype of column "Day".
    df1['Day_Name'] = df1['Day_Name'].astype('category')
    df1['Month'] = df1['Date'].dt.strftime('%B')  # Menggunakan %B untuk mendapatkan nama bulan
    df1['Year'] = df1['Date'].dt.year

    # ------------------------------------------------------------
    # # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Messages Sent per Day Over a Time Period</h2>", unsafe_allow_html=True)


    # Creating the Plotly Express line plot with color differentiation by Year
    fig = px.line(
        df1,
        x='Date',
        y='message_count',
        color='Year',  # Use 'color' for differentiation
    )

    # Updating the layout for better fit and appearance
    fig.update_layout(
        autosize=True,  # Set autosize to True
        xaxis_title='Date',
        yaxis_title='Message Count',
        margin=dict(l=20, r=20, t=50, b=20),  # Adjust margins
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)  # use_container_width=True ensures the chart uses full width

    ### Insight 3 - Total Number of Chats by Day, Month and Year ------------------------------------------------------------

    # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Message Count Analysis</h2>", unsafe_allow_html=True)

    # Displaying metrics using Streamlit
    col1, col2= st.columns(2)

    selected_variable = col1.selectbox("Select a variable to group by:", ('Day_Name', 'Month', 'Year'))
    color = col2.color_picker("Pick A Color for Bar", "#1f77b4")

    # Group by selected variable and sum message_count
    insight_2 = df1.groupby(selected_variable)['message_count'].sum().reset_index()

    # Sorting based on selected variable
    if selected_variable == 'Day_Name':
        insight_2 = insight_2.sort_values(by=selected_variable, key=lambda x: pd.Categorical(x, categories=['Monday', 'Tuesday',    'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True))
    elif selected_variable == 'Month':
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',    'December']
        insight_2 = insight_2.sort_values(by=selected_variable, key=lambda x: pd.Categorical(x, categories=month_order, ordered=True))

    # Plot bar chart using Plotly Express
    fig = px.bar(insight_2, x=selected_variable, y='message_count', labels={'message_count': 'Message Count'},  color_discrete_sequence=[color])

    # Customizing layout
    fig.update_layout(
        title=f'Grouped by {selected_variable}',
        title_x=0.5,  # Center the title
        xaxis_title=selected_variable,
        yaxis_title='Message Count',
        font=dict(size=14),  # Adjust font size
        autosize=True,  # Set autosize to True
        margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

    ### Insight 4 - Temporal Message Count Analysis ------------------------------------------------------------

    # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Temporal Message Count Analysis</h2>", unsafe_allow_html=True)

    # Selection box for grouping variable
    selected_variable = st.selectbox("Select a variable to group by:", ('Day_Name', 'Month'))

    # Grouping the data by Year and the selected variable
    df_temporal = df.groupby(['Year', selected_variable]).size().reset_index(name='Message Count')

    # Sorting based on selected variable
    if selected_variable == 'Day_Name':
        df_temporal = df_temporal.sort_values(by=selected_variable, key=lambda x: pd.Categorical(x, categories=['Monday', 'Tuesday',    'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True))
    elif selected_variable == 'Month':
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',    'December']
        df_temporal = df_temporal.sort_values(by=selected_variable, key=lambda x: pd.Categorical(x, categories=month_order,     ordered=True))

    # Temporal analysis using Plotly Express
    fig = px.line(df_temporal, x=selected_variable, y='Message Count', color='Year', labels={'Message Count': 'Message Count'},     title=f'Group by {selected_variable}')

    # Customizing layout
    fig.update_layout(
        title_x=0.4,  # Center the title
        xaxis_title=selected_variable,
        yaxis_title='Message Count',
        font=dict(size=14),  # Adjust font size
        autosize=True,
        margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # ### Insight 5 - Author Contribution Chats ------------------------------------------------------------

    # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Author Contribution in Group Chats Analysis</h2>", unsafe_allow_html=True)

    # Radio buttons to select the visualization
    option = st.radio("Select Visualization", ("Show Message Count", "Show Contribution Based on Words or Letters"))

    if option == "Show Message Count":
        # Compute author message counts
        df_author = df['Author'].value_counts().reset_index(name='Message Count').rename(columns={"index":  "Author"}).sort_values(by="Message Count", ascending=True)

        # Slider to select the number of bars to display
        show_bars = st.slider("Number of Bars to Display:", min_value=2, max_value=df_author.shape[0], value=10, key="number_of_bars")

        # Filter the DataFrame based on the slider value
        df_author_filtered = df_author.tail(show_bars)  # Select the top `show_bars` authors

        # Create horizontal bar chart using Plotly Express
        fig = px.bar(df_author_filtered, x='Message Count', y='Author',
                     labels={'Message Count': 'Message Count', 'Author': 'Author'},
                     orientation='h',  # Set orientation to horizontal
                     title='Author Contribution Number of Chats in Group', color_discrete_sequence=[color])

        # Customizing layout
        fig.update_layout(
            title_x=0.4,  # Center the title
            xaxis_title='Message Count',
            yaxis_title='Author',
            font=dict(size=14),  # Adjust font size
            autosize=True,
            margin=dict(l=20, r=20, t=50, b=20)  # Adjust margins
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    elif option == "Show Contribution Based on Words or Letters":
        # Filter selection
        Select_type = st.selectbox("Select type of contribution", ["Word's", "Letter's"])

        # Data processing based on filter
        df_author_based_on_words = df.groupby('Author')[Select_type].sum().reset_index(name=f'{Select_type} Count').sort_values(by=f'{Select_type} Count', ascending=True).reset_index(drop=True)

        # Slider to select the number of bars to display
        show_bars = st.slider("Number of Bars to Display:", min_value=2, max_value=df_author_based_on_words.shape[0], value=10,     key="number_of_bars")

        # Filter the DataFrame based on the slider value
        df_author_filtered = df_author_based_on_words.tail(show_bars)  # Select the top `show_bars` authors

        # Create horizontal bar chart using Plotly Express
        fig = px.bar(df_author_filtered, x=f'{Select_type} Count', y="Author",
                     labels={f'{Select_type} Count': f'{Select_type} Count', 'Author': 'Author'},
                     orientation='h',  # Set orientation to horizontal
                     title=f"Author Contribution Based on {Select_type} Count", color_discrete_sequence=[color])

        # Customizing layout
        fig.update_layout(
            title_x=0.4,  # Center the title
            xaxis_title=f'{Select_type} Count',
            yaxis_title='Author',
            font=dict(size=14),  # Adjust font size
            autosize=True,
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig)

    ### Insight 6 - Total Number of Each Emoji ------------------------------------------------------------

    # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Emoji Usage Counts</h2>", unsafe_allow_html=True)


    # Compute the total list of emojis
    total_emojis_list = list([a for b in df["Emoji's"] for a in b])
    total_emojis = len(set(total_emojis_list))

    # Compute emoji counts
    emoji_dict = dict(Counter(total_emojis_list))
    emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)

    # Create a DataFrame for the emojis
    emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])

    # Slider to select the number of bars to display
    show_emoji = st.slider("Number of Emoji:", min_value=2, max_value=emoji_df.shape[0], value=10, key="number_of_emoji_shows")

    # Selecting the top ten emojis
    top_emojis = emoji_df.head(show_emoji)

    # Create a bar chart using Plotly Express
    fig = px.bar(top_emojis, x='emoji', y='count', title=f'Top {show_emoji} Emoji Usage Count', color_discrete_sequence=[color])

    # Customizing layout
    fig.update_layout(
        title_x=0.4,  # Center the title
        autosize=True,
        xaxis_title='Emoji',
        yaxis_title='Count',
        font=dict(size=14),
    )

    # Show the bar chart for the top ten emojis
    st.plotly_chart(fig)


    ### Insight 6 - Total Number of Each Emoji ------------------------------------------------------------

    # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Most Active Hours</h2>", unsafe_allow_html=True)

    # Create a copy of the DataFrame and add a message_count column
    df3 = df.copy()
    df3['message_count'] = 1

    # Convert 'Time' column to datetime, then extract hour
    df3['Time'] = pd.to_datetime(df3['Time'], format='%H:%M:%S').dt.time
    df3['hour'] = pd.to_datetime(df3['Time'].astype(str), format='%H:%M:%S').dt.hour

    # Group by hour and sum message counts
    grouped_by_time = df3.groupby('hour')['message_count'].sum().reset_index().sort_values(by='hour')

    # Beautifying Default Styles using Seaborn
    sns.set_style("whitegrid")
    sns.set_palette("viridis")

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(16, 8))

    # PLOT: grouped by hour
    sns.barplot(x='hour', y='message_count', data=grouped_by_time, ax=ax)

    # Adding labels and title
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Message Count', fontsize=12)

    # Rotating x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)

    # Adding grid for better visualization
    ax.yaxis.grid(True)

    # Adding value annotations on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Adjust plot margins
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
    ### Insight 7 - Extract basic statistics based on specific authors --------------------------------------------------------
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    from nltk.stem import PorterStemmer
    import string
    import random
    # Download stopwords if not already downloaded
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    
    
    # Streamlit title and selection widget
    st.markdown("<h2 style='text-align: center;'>Extract basic statistics based on specific authors</h2>", unsafe_allow_html=True)
    
    # Mendapatkan semua penulis unik
    all_authors = df['Author'].unique()
    
        # Pilihan dropdown untuk penulis
    selected_author = st.selectbox('Pilih Penulis', all_authors)
    
    # Filter data berdasarkan penulis yang dipilih
    req_df_spesific_author = df[df["Author"] == selected_author]

    # Fungsi untuk mengubah teks menjadi huruf kecil
    def to_lowercase(df, column):
        df[column] = df[column].apply(lambda x: x.lower() if isinstance(x, str) else x)
        return df

    # Fungsi untuk menghapus angka dari string
    def remove_numbers(df, column):
        df[column] = df[column].apply(lambda x: regex.sub(r'\d+', '', x) if isinstance(x, str) else str(x))
        return df

    # Fungsi untuk menghapus tanda baca dari string
    def remove_punctuation(df, column):
        df[column] = df[column].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)) if isinstance(x, str) else x)
        return df

    # Fungsi untuk menghapus stopwords
    def remove_stopwords(text, stop_words):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    # Menggabungkan stopwords dari NLTK dan Sastrawi
    def get_combined_stopwords():
        stop_words_nltk = set(stopwords.words('indonesian'))
        stop_factory = StopWordRemoverFactory()
        more_stopwords = [
            'yg','dengan', 'ia','bahwa','media', 'omitted','oleh','dan','ini','itu',
            'indonesia','ada','aku','untuk','tapi','mau','jadi', 'aja', 'nya', 'tak', 
            'gak', 'apa', 'kalo', 'pak', 'sama', 'semua', 'gue', 'kan','bukan', 'url', 
            'baru', 'anak', 'satu', 'punya', 'udah', 'kau', 'utk', 'kata', 'tau', 'biar', 
            'lebih', 'naik', 'cuma', 'mana', 'siapa', 'bilang', 'haru', 'menjadi','orang',
            'sih','amp','kok','kalau', 'jangan', 'gua', 'buat', 'lah', 'kok', 'banyak', 
            'kamu','jangan', 'dulu', 'malah', 'emang', 'tuh', 'teru', 'sekarang', 'bang', 
            'lalu', 'hidup','memang', 'samp', 'klo', 'pernah', 'masuk', 'nih', 'banget', 
            'pale', 'manusia', 'minta', 'makan', 'mungkin', 'padah'
        ]
        stop_words_sastrawi = stop_factory.get_stop_words() + more_stopwords
        return stop_words_nltk.union(stop_words_sastrawi)

    # Fungsi utama untuk preprocessing teks
    def preprocess_text(df, column):
        stop_words_combined = get_combined_stopwords()
        df = to_lowercase(df, column)
        df = remove_numbers(df, column)
        df = remove_punctuation(df, column)
        df[column] = df[column].apply(lambda x: remove_stopwords(x, stop_words_combined))
        return df

    # # Menambahkan spinner untuk menandakan proses sedang berjalan
    # with st.spinner('Processing Text...'):
    #     # Preprocess text in 'Message' column
    df_spesific_author_after_processing = preprocess_text(req_df_spesific_author.copy(), 'Message')

    
    # Displaying metrics using Streamlit
    col1, col2= st.columns(2)

    with col1:
        # Daftar emoji
        emoji_list = ["🔥", "😊", "🎉", "🚀", "💡", "🌟", "👍"]

        # Pilih emoji secara acak
        emot = random.choice(emoji_list)
        
        # Mendefinisikan HTML untuk top bar ------------------------------------------------------------
        top_bar = f"""
        <div style="background-color:#333; padding:5px;border-radius: 10px;">
            <h3 style="color:white; text-align:center;font-size: 20px">{emot} Statistik dari {selected_author} {emot}</h3>
        </div>
        """

        # Menampilkan top bar sebagai komponen HTML
        st.markdown(top_bar, unsafe_allow_html=True)
        st.divider()
        st.write('Total Pesan Dikirim:', req_df_spesific_author.shape[0])
        words_per_message = (np.sum(req_df_spesific_author["Word's"])) / req_df_spesific_author.shape[0]
        w_p_m = ("%.3f" % round(words_per_message, 2))
        st.write('Rata-rata Kata per Pesan:', w_p_m)
        spesific_author_media = sum(req_df_spesific_author["Media_Count"])
        st.write('Total Media Pesan Dikirim:', spesific_author_media)
        spesific_author_links = sum(req_df_spesific_author["Url_Count"])
        st.write('Total Tautan Dikirim:', spesific_author_links)
        percentage_of_activity = (req_df_spesific_author.shape[0]/total_messages) * 100
        st.write('Presentase Keaftifan Pengiriman Pesan:', spesific_author_links, "%")
        
    with col2:
        # Menggabungkan semua teks dari kolom "JAWABAN" dalam satu string
        pesan_combined = ' '.join(df_spesific_author_after_processing['Message'].astype(str).values)
        word_seperated_pesan = word_tokenize(pesan_combined)
        kemunculan_word = nltk.FreqDist(word_seperated_pesan)
        
        show_bars_kemunculan = st.number_input("Number of Words:", min_value=3, max_value=len(kemunculan_word), value=10, step=1, key="show_bars_kemunculan")

        # Mendapatkan kata dan frekuensi kemunculan kata yang paling umum
        kata_kemunculan_word = kemunculan_word.most_common(show_bars_kemunculan)
        kata_umum, frekuensi_umum = zip(*kata_kemunculan_word)

        # Membuat DataFrame untuk plot
        df_plot = pd.DataFrame({'Kata': kata_umum, 'Frekuensi': frekuensi_umum})    
    
        # Membuat plot dengan Plotly Express
        fig = px.bar(df_plot, x='Kata', y='Frekuensi', title=f'Kata yang sering digunakan oleh {selected_author} {emot}',
                     labels={'Kata': 'Kata', 'Frekuensi': 'Frekuensi'}, color='Frekuensi', color_continuous_scale=px.colors.sequential.Plasma)    
    
        # Menyesuaikan tampilan plot
        fig.update_layout(xaxis_tickangle=-45, xaxis_title='Kata', yaxis_title='Frekuensi',
                          title_font_size=20, xaxis_tickfont_size=12, yaxis_tickfont_size=12)
        fig.update_traces(texttemplate='%{y}', textposition='outside')    
    
        # Menampilkan plot di Streamlit
        st.plotly_chart(fig)
    
    ### Insight 8 - Wordcloud for Text --------------------------------------------------------

    # Membuat WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma', max_words=100).generate(pesan_combined)
    
    title_wordcloud = f"""
        <h3 style="color:black; text-align:center;">Wordcloud for {selected_author} {emot}</h3>
    """
    
    # Menampilkan top bar sebagai komponen HTML
    st.markdown(title_wordcloud, unsafe_allow_html=True)



    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)



    
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