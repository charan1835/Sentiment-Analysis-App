import streamlit as st
import joblib
import os

# --- Constants ---
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# --- Caching and Model Loading ---
@st.cache_resource
def load_model_and_vectorizer():
    """Load the sentiment model and vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

def main_page():
    """Defines the layout and logic for the main Sentiment Analyzer page."""
    # --- Custom CSS ---
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

            body {
                background-color: #111111; /* Graphite Black */
                color: #EAEAEA; /* Light Grey Text */
                font-family: 'Poppins', sans-serif;
            }
            .main {
                background: #1E1E1E; /* Darker Graphite */
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid #333333;
                box-shadow: 0px 8px 30px rgba(0,0,0,0.7);
            }
            h1 {
                background: -webkit-linear-gradient(45deg, #39FF14, #23a6d5); /* Mint to Blue */
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5rem !important;
                font-weight: 700;
            }
            .stTextArea textarea {
                background-color: #2C2C2C !important;
                color: #EAEAEA !important;
                border-radius: 10px;
                border: 1px solid #444444;
            }
            .stButton button {
                background: linear-gradient(90deg, #39FF14, #00b39f); /* Mint Green Gradient */
                color: white;
                border-radius: 10px;
                padding: 0.6rem 1.5rem;
                font-weight: 600;
                border: none;
                transition: 0.3s;
            }
            .stButton button:hover {
                transform: scale(1.05);
                box-shadow: 0px 4px 20px rgba(57, 255, 20, 0.3);
            }
            .result-box {
                background-color: rgba(44, 44, 44, 0.6);
                padding: 1rem;
                border-radius: 10px;
                margin-top: 1.5rem;
                border-left: 5px solid #39FF14;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🧠 Sentiment Analysis App")
    st.subheader("Analyze social media comments with a pre-trained ML model.")

    if model is None or vectorizer is None:
        st.error("🔴 **Error:** Model or vectorizer files not found. Please make sure `sentiment_model.pkl` and `vectorizer.pkl` are in the same directory.")
    else:
        user_input = st.text_area("💬 Type your comment here:", height=100)
        if st.button("🔍 Analyze Sentiment"):
            if user_input.strip():
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0].capitalize()
                emoji_map = {"Positive": "😄", "Neutral": "😐", "Negative": "😞"}
                emoji = emoji_map.get(prediction, "🤔")
                st.markdown(f"""<div class="result-box">
                                <h3>🔹 Sentiment: <strong>{prediction} {emoji}</strong></h3>
                            </div>""", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please type a comment before analyzing!")
    st.markdown("</div>", unsafe_allow_html=True)

# Define pages for navigation
pages = [
    st.Page(main_page, title="Sentiment Analyzer", icon="🧠", default=True),
    st.Page("pages/1_About.py", title="About the App", icon="ℹ️")
]

# Create and run navigation
pg = st.navigation(pages)
pg.run()
