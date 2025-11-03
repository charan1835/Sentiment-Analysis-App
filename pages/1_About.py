import streamlit as st
import os
import sys

# Add the parent directory to the path to import the styling function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the shared styling function
from app import main_page_styles

# --- Page Config ---
st.set_page_config(page_title="About", page_icon="ℹ️", layout="centered")

# --- Custom CSS for consistency ---
main_page_styles() # Apply shared styles

st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("ℹ️ About This App")

st.markdown("""
This application is an interactive tool for sentiment analysis. It uses a machine learning model to classify text into **Positive**, **Negative**, or **Neutral** categories.
""")

st.divider()

st.header("🎯 Project Goals")
st.markdown("""
This project was built to demonstrate a simple, end-to-end MLOps workflow:
-   **Model Training:** A sentiment analysis model was trained and evaluated in a separate environment (e.g., a Jupyter Notebook).
-   **Model Serialization:** The trained model and its associated vectorizer were saved to disk using `joblib`.
-   **Deployment:** The serialized artifacts are loaded into this Streamlit web application, which provides an interactive user interface for real-time predictions.
-   **Containerization (Optional):** The app is ready to be containerized with Docker for scalable and reproducible deployment.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("⚙️ How It Works")
    st.markdown("""
    The sentiment analysis pipeline consists of two key components:
    -   **TF-IDF Vectorizer:** Converts raw text into numerical vectors. It highlights words that are important to a comment relative to their frequency across all comments.
    -   **Logistic Regression Model:** A robust and interpretable classification algorithm that predicts the sentiment based on the vectorized text. It was trained on a labeled dataset of social media comments, achieving over 80% accuracy.
    """)

with col2:
    st.header("💻 Tech Stack")
    st.markdown("""
    -   **Backend & ML:** Python, Scikit-learn, Joblib
    -   **Frontend:** Streamlit
    -   **Data Handling:** Pandas, NumPy
    -   **Deployment:** Streamlit Community Cloud (or Docker)
    """)

st.divider()

with st.expander("🚀 Run This App Locally"):
    st.code("""
    # 1. Clone the repository
    git clone https://github.com/charan1835/sentiment-analysis-app.git
    cd sentiment-analysis-app

    # 2. Install dependencies
    pip install -r requirements.txt

    # 3. Run the Streamlit app
    streamlit run app.py
    """, language="bash")

st.divider()

st.markdown("""
---
*This application is for demonstration purposes. The model's performance may vary on text from different domains.*
<br>
*Created by Charan.*
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)