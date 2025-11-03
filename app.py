import streamlit as st
import joblib
import os
import numpy as np
import re

# --- Constants ---
MODEL_PATH = "sentiment_models.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

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

def _softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_word_contributions(text, model, vectorizer):
    """
    Analyzes the contribution of each word to the sentiment prediction.
    Returns a dictionary of words and their contribution scores.
    """
    if 'positive' not in model.classes_ or 'negative' not in model.classes_:
        # This handles cases where the model classes are not as expected
        return {}, None

    # Get the class indices for positive and negative sentiment
    positive_class_index = list(model.classes_).index('positive')
    negative_class_index = list(model.classes_).index('negative')

    # Create a mapping from feature index to word
    feature_names = vectorizer.get_feature_names_out()
    
    # Get coefficients for positive and negative classes
    pos_coeffs = model.coef_[positive_class_index]
    neg_coeffs = model.coef_[negative_class_index]

    # Create a dictionary mapping words to their influence score
    # Score = (positive coefficient - negative coefficient)
    word_scores = {word: pos_coeffs[i] - neg_coeffs[i] for word, i in vectorizer.vocabulary_.items()}
    
    return word_scores, model.intercept_

def highlight_text(text, word_scores, threshold):
    """
    Highlights words in the text based on their contribution scores.
    """
    highlighted_html = ""
    # Use regex to split text while preserving punctuation and spaces
    tokens = re.findall(r"(\w+|[^\w\s])(\s*)", text) # Capture words, punctuation, and trailing spaces
    
    for token, space in tokens:
        word = token.lower().strip()
        score = word_scores.get(word, 0)
        
        if score > threshold: # Strong positive contribution
            highlighted_html += f'<span style="background-color: #28a745; color: white; padding: 2px 4px; border-radius: 4px;">{token}</span>{space}'
        elif score < -threshold: # Strong negative contribution
            highlighted_html += f'<span style="background-color: #dc3545; color: white; padding: 2px 4px; border-radius: 4px;">{token}</span>{space}'
        else:
            highlighted_html += f"{token}{space}"
            
    return highlighted_html.strip()

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
            .probability-bar-container {
                display: flex;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            .probability-label {
                width: 80px; /* Fixed width for labels */
                margin-right: 10px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🧠 Sentiment Analysis App")
    st.subheader("Analyze social media comments with a pre-trained ML model.")

    # --- Example Comments ---
    st.markdown("<p style='margin-top: 1rem; margin-bottom: 0.5rem;'>👇 Or try one of these examples:</p>", unsafe_allow_html=True)
    examples = {
        "Strongly Positive": "This is the best thing I've ever seen! Absolutely amazing. 10/10!",
        "Positive Service": "The customer service was outstanding and very friendly.",
        "Neutral": "It does the job. Nothing more, nothing less.",
        "Mixed/Neutral": "The food was average, but the crew was polite and professional.",
        "Slightly Negative": "The delivery was a bit late, which was disappointing.",
        "Strongly Negative": "A terrible product, I would not recommend it to anyone at all."
    }

    # Initialize session state for text_area if it doesn't exist
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # Display example buttons in a grid layout
    cols = st.columns(3)
    example_items = list(examples.items())
    for i, col in enumerate(cols):
        with col:
            if i*2 < len(example_items):
                label, text = example_items[i*2]
                if st.button(label, help=text, use_container_width=True):
                    st.session_state.user_input = text
            if i*2 + 1 < len(example_items):
                label, text = example_items[i*2 + 1]
                if st.button(label, help=text, use_container_width=True):
                    st.session_state.user_input = text

    st.divider()

    if model is None or vectorizer is None:
        st.error("🔴 **Error:** Model or vectorizer files not found. Please make sure `sentiment_models.pkl` and `tfidf_vectorizer.pkl` are in the same directory.")
    else:
        user_input = st.text_area("💬 Type or select a comment:", height=100, key="user_input")
        
        # Action buttons in columns
        col1, col2 = st.columns([3, 1]) # Give more space to the Analyze button
        analyze_button = col1.button("🔍 Analyze Sentiment", use_container_width=True)
        if col2.button("🧹 Clear", use_container_width=True):
            st.session_state.user_input = ""
            st.rerun()

        if analyze_button:
            if user_input.strip():
                X = vectorizer.transform([user_input])
                prediction = model.predict(X)[0].capitalize()
                
                # Use decision_function and softmax as a fallback for predict_proba
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)[0]
                else:
                    # For LinearSVC and other models without predict_proba
                    decision_scores = model.decision_function(X)[0]
                    probabilities = _softmax(decision_scores)

                
                emoji_map = {"Positive": "😄", "Neutral": "😐", "Negative": "😞"}
                emoji = emoji_map.get(prediction, "🤔")
                
                with st.container(border=True):
                    st.subheader(f"Predicted Sentiment: {prediction} {emoji}")
                    st.write("Confidence Scores:")
                    for i, class_label in enumerate(model.classes_):
                        st.progress(probabilities[i], text=f"{class_label.capitalize()}: {probabilities[i]:.2%}")
                    
                    st.divider()
                    
                    col1, col2 = st.columns([2,1])
                    with col1:
                        st.write("💡 **Key Word Contributions**")
                        st.markdown("""
                            <small>Words that strongly influenced the prediction.</small><br>
                            <span style="background-color: #28a745; color: white; padding: 1px 3px; border-radius: 3px;">Positive</span> 
                            <span style="background-color: #dc3545; color: white; padding: 1px 3px; border-radius: 3px;">Negative</span>
                        """, unsafe_allow_html=True)
                    with col2:
                        threshold = st.slider("Highlight Sensitivity", 0.1, 2.0, 0.5, 0.1, help="Lower values highlight more words.")

                    word_scores, _ = get_word_contributions(user_input, model, vectorizer)
                    highlighted_output = highlight_text(user_input, word_scores, threshold)
                    st.markdown(f"<div style='margin-top: 1rem; font-size: 1.1rem;'>{highlighted_output}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please type a comment before analyzing!")
    st.markdown("</div>", unsafe_allow_html=True)

# Define pages for navigation
pages = [
    st.Page(main_page, title="Sentiment Analyzer", icon="🧠", default=True),
    st.Page("pages/2_Bulk_Analysis.py", title="Bulk Analysis", icon="📂"),
    st.Page("pages/1_About.py", title="About the App", icon="ℹ️"),
]

# Create and run navigation
pg = st.navigation(pages)
pg.run()
