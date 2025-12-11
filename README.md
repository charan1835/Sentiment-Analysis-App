# 🧠 Sentiment Analysis Web App

A powerful and interactive **Machine Learning application** built with Streamlit that classifies text sentiment as **Positive**, **Negative**, or **Neutral**. 

This application demonstrates an end-to-end MLOps workflow, from model training (Logistic Regression + TF-IDF) to a user-friendly deployment.

## 🚀 Features

### 1. 🔍 Instant Sentiment Analysis
- **Real-time Prediction**: Type any sentence to instantly see if it's Positive, Negative, or Neutral.
- **Visual Feedback**: Results are color-coded (Teal for Positive, Red for Negative) with appropriate emojis.
- **Word Highlighting**: Understand *why* a prediction was made. The app highlights words that strongly influenced the sentiment (e.g., "amazing" in green, "terrible" in red).
- **Confidence Scores**: See the model's certainty for each class (e.g., 95% Positive).

### 2. 📂 Bulk Analysis
- **Batch Processing**: Paste a list of comments (one per line) to analyze them all at once.
- **Data Visualization**: View a bar chart of sentiment distribution within your bulk data.
- **Export Results**: Download the complete analysis results (including confidence scores) as a CSV file.

### 3. 📤 File Upload
- **Large Scale Analysis**: Upload `.txt` or `.csv` files containing hundreds or thousands of comments.
- **Automatic parsing**: Handles text files line-by-line and detects text columns in CSVs.
- **Comprehensive Reports**: Generates pie charts and word clouds to summarize the uploaded dataset.

### 4. ☁️ Word Cloud Generation
- Visualize the most frequent terms in your text data to spot common themes or complaints instantly.

### 5. 🕒 Session History
- Keep track of your recent analyses in the sidebar, so you don't lose context while experimenting.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) (Python-based web framework)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Logistic Regression)
- **NLP**: TF-IDF Vectorization for text feature extraction.
- **Utilities**: Pandas, NumPy, Joblib (for model serialization), Matplotlib/WordCloud (for visualization).

## 📂 Project Structure

```
├── app.py                 # Main application entry point (Single Analysis)
├── utils.py               # Helper logic (softmax, history management)
├── view_utils.py          # UI components (styles, plots, sidebar)
├── pages/
│   ├── 1_About.py         # Project information and documentation
│   ├── 2_Bulk_Analysis.py # Logic for pasting and analyzing multiple lines
│   └── 3_File_Upload.py   # Logic for file-based analysis
├── sentiment_model (2).pkl      # Trained Logistic Regression model
├── tfidf_vectorizer (1).pkl     # Fitted TF-IDF Vectorizer
├── requirements.txt       # Python dependencies
└── css/ (Optional)        # Custom styles injected via view_utils.py
```

## 🧩 How It Works

1.  **Preprocessing**: The text input is cleaned and tokenized.
2.  **Vectorization**: A **TF-IDF Vectorizer** converts the text into numerical numbers, weighing unique words higher than common ones (like "the" or "is").
3.  **Prediction**: The pre-trained **Logistic Regression** model calculates a score for each sentiment class.
4.  **Interpretation**: The app extracts the learned coefficients from the model to show exactly which words contributed to the positive or negative score.

## 💻 Setup & Installation

Follow these steps to run the app locally on your machine.

**Prerequisites**: Python 3.8+ installed.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/charan1835/sentiment-analysis-app.git
    cd sentiment-analysis-app
    ```

2.  **Install Dependencies**
    It's recommended to create a virtual environment first.
    ```bash
    # Create virtual environment (optional but recommended)
    python -m venv .venv
    
    # Activate it:
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate

    # Install libraries
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```
    The app will open automatically in your browser at `http://localhost:8501`.

## 🤝 Contributing

Contributions are welcome! If you find a bug or want to suggest a feature:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature-name`).
3.  Commit your changes.
4.  Push to the branch and create a Pull Request.

---
*Created by Charan.*
