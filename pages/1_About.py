import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="About", page_icon="ℹ️", layout="centered")

# --- Custom CSS for consistency ---
st.markdown("""
    <style>

        body {
            background-color: #111111; /* Graphite Black */
            color: #EAEAEA; /* Light Grey Text */
            font-family: sans-serif;
        }
        .main-about {
            background: #1E1E1E; /* Darker Graphite */
            padding: 2rem;
            border-radius: 20px;
            border: 1px solid #333333;
            box-shadow: 0px 8px 30px rgba(0,0,0,0.7);
        }
        h1, h2 {
            color: #39FF14; /* Solid Mint Green */
            font-weight: 600;
        }
        .stCodeBlock {
            background-color: #2C2C2C !important;
            border-radius: 10px;
        }
        ul {
            list-style-position: inside;
            padding-left: 0;
        }
        li {
            margin-bottom: 0.5rem;
        }
        .st-emotion-cache-1h9us21 p {
            font-size: 1.1rem;
        }
        .st-emotion-cache-1h9us21 {
            border-color: #444444;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-about'>", unsafe_allow_html=True)

st.title("ℹ️ About This App")

st.markdown("""
This is a simple Machine Learning web application that predicts the sentiment of a given text. 
The model classifies comments as **Positive**, **Negative**, or **Neutral**.
""")

st.divider()

col1, col2 = st.columns([1.2, 0.8])

with col1:
    st.header("⚙️ How It Works")
    st.markdown("""
    The application uses a pre-trained machine learning model to perform sentiment analysis. The pipeline consists of two main components:
    -   **TF-IDF Vectorizer:** This converts the input text into a numerical format (vectors) that the model can understand. It focuses on the importance of words within the document.
    -   **Logistic Regression:** A robust and efficient classification model that takes the numerical vectors and predicts the sentiment. The model was trained on a labeled dataset of social media comments and achieved over 80% accuracy.
    """)

with col2:
    st.header("💻 Tech Stack")
    st.markdown("""
    -   **Python**
    -   **Streamlit**
    -   **Scikit-learn**
    -   **Joblib**
    """)

with st.expander("🚀 How to Run Locally"):
    st.write("Follow these steps to run the app on your own machine:")
    st.code("""
    # 1. Clone the repository
    git clone https://github.com/charan1835/sentiment-analysis-app.git
    cd sentiment-analysis-app

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Run the Streamlit app
    streamlit run app.py
    """, language="bash")

st.markdown("</div>", unsafe_allow_html=True)