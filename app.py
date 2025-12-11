import streamlit as st
import joblib
import os
import numpy as np
import re
from utils import softmax, add_to_history
from view_utils import main_page_styles, show_history_sidebar

# --- Project Root Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
MODEL_FILENAME = "sentiment_model (2).pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer (1).pkl"

# --- Caching and Model Loading ---
@st.cache_resource
def load_model_and_vectorizer():
    """Load the sentiment model and vectorizer from disk."""
    try:
        # Try to find the model and vectorizer files in the current directory or parent directory
        model_path = None
        vectorizer_path = None
        
        # Check in current directory
        if os.path.isfile(os.path.join(BASE_DIR, MODEL_FILENAME)) and os.path.isfile(os.path.join(BASE_DIR, VECTORIZER_FILENAME)):
            model_path = os.path.join(BASE_DIR, MODEL_FILENAME)
            vectorizer_path = os.path.join(BASE_DIR, VECTORIZER_FILENAME)
        # Check in parent directory (for Streamlit Cloud deployment)
        elif os.path.isfile(os.path.join(os.path.dirname(BASE_DIR), MODEL_FILENAME)) and \
             os.path.isfile(os.path.join(os.path.dirname(BASE_DIR), VECTORIZER_FILENAME)):
            model_path = os.path.join(os.path.dirname(BASE_DIR), MODEL_FILENAME)
            vectorizer_path = os.path.join(os.path.dirname(BASE_DIR), VECTORIZER_FILENAME)
        else:
            st.error(f"Model or vectorizer files not found. Please make sure {MODEL_FILENAME} and {VECTORIZER_FILENAME} are in the project's root directory.")
            st.error(f"Current directory: {os.getcwd()}")
            st.error(f"Files in current directory: {os.listdir('.')}")
            return None, None
            
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
        
    except Exception as e:
        st.error(f"Error loading the sentiment analysis model: {str(e)}")
        return None, None

model, vectorizer = load_model_and_vectorizer()

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
        
        if score > threshold: # Strong positive contribution (Teal)
            highlighted_html += f'<span style="background-color: #14B8A6; color: white; padding: 2px 4px; border-radius: 4px;">{token}</span>{space}'
        elif score < -threshold: # Strong negative contribution (Rose Red)
            highlighted_html += f'<span style="background-color: #F43F5E; color: white; padding: 2px 4px; border-radius: 4px;">{token}</span>{space}'
        else:
            highlighted_html += f"{token}{space}"
            
    return highlighted_html.strip()

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

def main_page():
    """Defines the layout and logic for the main Sentiment Analyzer page."""
    # Apply shared styles
    main_page_styles()

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🧠 Sentiment Analysis App")
    st.subheader("Analyze social media comments with a pre-trained ML model.")

    # Initialize session state for text_area if it doesn't exist
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # --- Example Comments in an Expander ---
    with st.expander("👇 Try an example"):
        examples = {
            "Strongly Positive": "This is the best thing I've ever seen! Absolutely amazing.",
            "Positive Service": "The customer service was outstanding and very friendly.",
            "Neutral": "It does the job. Nothing more or less.",
            "Mixed/Neutral": "The food was average, but the crew was polite and professional.",
            "Slightly Negative": "The delivery was a bit late, which was disappointing.",
            "Strongly Negative": "A terrible product, I would not recommend it to anyone at all."
        }

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
        def clear_input():
            st.session_state.user_input = ""
            st.rerun()

        col2.button("🧹 Clear", use_container_width=True, on_click=clear_input)

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
                    probabilities = softmax(decision_scores)

                
                emoji_map = {"Positive": "😄", "Neutral": "😐", "Negative": "😞"}
                emoji = emoji_map.get(prediction, "🤔")
                
                # Add to history
                add_to_history(user_input, prediction)
                
                # --- Display Results with Contextual Feedback ---
                with st.container(border=True):
                    if prediction == "Positive":
                        st.subheader(f"Predicted Sentiment: Positive {emoji}")
                        st.success("This comment reflects positive sentiment, indicating favorable feedback.", icon="✅")
                        st.balloons()
                    elif prediction == "Negative":
                        st.subheader(f"Predicted Sentiment: Negative {emoji}")
                        st.warning("This comment conveys negative sentiment. This may highlight an area for review or improvement.", icon="⚠️")
                    else: # Neutral
                        st.subheader(f"Predicted Sentiment: Neutral {emoji}")
                        st.info("This comment is classified as neutral, likely representing a factual statement or objective feedback.", icon="✍️")

                    st.write("Confidence Scores:")
                    for i, class_label in enumerate(model.classes_):
                        st.progress(probabilities[i], text=f"{class_label.capitalize()}: {probabilities[i]:.2%}")
                    
                    st.divider()
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("💡 **Key Word Contributions**")
                        st.markdown("""
                            <small>Words that influenced the prediction.</small><br>
                            <span style="background-color: #14B8A6; color: white; padding: 1px 3px; border-radius: 3px;">Positive</span> 
                            <span style="background-color: #F43F5E; color: white; padding: 1px 3px; border-radius: 3px;">Negative</span>
                        """, unsafe_allow_html=True)
                    with col2:
                        threshold = st.slider("Highlight Sensitivity", 0.1, 2.0, 0.5, 0.1, help="Lower values highlight more words.")

                    word_scores, _ = get_word_contributions(user_input, model, vectorizer)
                    highlighted_output = highlight_text(user_input, word_scores, threshold)
                    st.markdown(f"<div style='margin-top: 1rem; font-size: 1.1rem;'>{highlighted_output}</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Please type a comment before analyzing!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Show history sidebar
    show_history_sidebar()

# Define pages for navigation
pages = [
    st.Page(main_page, title="Sentiment Analyzer", icon="🧠", default=True),
    st.Page("pages/2_Bulk_Analysis.py", title="Bulk Analysis", icon="📂"),
    st.Page("pages/3_File_Upload.py", title="File Upload Analysis", icon="📤"),
    st.Page("pages/1_About.py", title="About the App", icon="ℹ️"),
]

# Create and run navigation
pg = st.navigation(pages)
pg.run()
