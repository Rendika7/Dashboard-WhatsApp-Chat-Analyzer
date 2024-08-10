# Dashboard WhatsApp Chat Analyzer with Streamlit
---

![WhatsApp Chat Analyzer](https://download.logo.wine/logo/WhatsApp/WhatsApp-Logo.wine.png)

---
Welcome to my Personal Dashboard, a powerful tool to automatically analyze WhatsApp Group Chat dialogues! Just upload the `.txt` file of your WhatsApp conversations and enjoy a deep understanding of your interactions. Let's start your smart analysis now! 🚀

## Features

This dashboard includes three main pages:

1. **Homepage**
    - Upload the `.txt` file from your WhatsApp group chat export.
    - Convert the `.txt` file to a dataframe for further analysis.

2. **WhatsApp Analysis**
    - Perform exploratory data analysis (EDA) on the uploaded chat data.
    - Simple exploration of chat data to uncover insights and patterns.

3. **Sentiment Analysis**
    - Determine the sentiment of each message sent by users in the chat.
    - Sentiments are categorized into Negative, Positive, and Neutral.
    - Uses BERT Transformers model `mdhugol/indonesia-bert-sentiment-classification` for sentiment analysis (Note: the accuracy might not be very high).

### Struktur Repositori
```
├── .ipynb_checkpoints
├── .streamlit
├── __pycache__
├── Data
│   └── WhatsApp Chat with PETARUNX DATA🤼_♂️.txt
├── pages
│   ├── 2_📈_WhatsappAnalysis.py
│   └── 3_🦊_SentimentAnalysis.py
├── Source
├── Wordcloud Masking
├── 1_🎈_Homepage.py
├── Documentation.txt
├── README.md
├── requirements.txt
├── temp_file.txt
└── Whatsapp Analysis Python.ipynb
```

## Installation

To get started with the dashboard, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Rendika7/dashboard-whatsapp-chat-analyzer.git
    ```
2. Navigate to the project directory:
    ```bash
    cd dashboard-whatsapp-chat-analyzer
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the web browser and go to `http://localhost:8501`.
2. On the **Homepage**, upload your WhatsApp group chat `.txt` file.
3. Navigate to the **WhatsApp Analysis** page to perform exploratory data analysis on the uploaded chat data.
4. Go to the **Sentiment Analysis** page to analyze the sentiment of each message in the chat.

## Example

Here's a brief example of how to use the dashboard:

1. **Homepage**: Upload your WhatsApp chat export file.
    - ![Homepage](https://github.com/Rendika7/Dashboard-WhatsApp-Chat-Analyzer/blob/main/Source/Homepages.png) 

2. **WhatsApp Analysis**: View the exploratory data analysis of your chat data.
    - ![WhatsApp Analysis](https://github.com/Rendika7/Dashboard-WhatsApp-Chat-Analyzer/blob/main/Source/WhatsappAnalysis.png)

3. **Sentiment Analysis**: Analyze the sentiment of each message in your chat.
    - ![Sentiment Analysis](https://github.com/Rendika7/Dashboard-WhatsApp-Chat-Analyzer/blob/main/Source/SentimentAnalysis.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://www.streamlit.io/)
- [BERT Transformers](https://github.com/mdhugol/indonesia-bert-sentiment-classification)
- Special thanks to the open-source community for their invaluable tools and resources.

---

Happy Analyzing! 🎉
