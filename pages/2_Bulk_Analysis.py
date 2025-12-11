import streamlit as st
import pandas as pd
import os
import sys
import joblib
from view_utils import main_page_styles, generate_wordcloud_plot, show_history_sidebar
from utils import add_to_history

# --- Project Root Directory (Match app.py logic) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
MODEL_FILENAME = "sentiment_model (2).pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer (1).pkl"

# Define model and vectorizer at module level
model = None
vectorizer = None
MODEL_LOADED = False

# --- Load Model and Vectorizer (Using same logic as app.py) ---
try:
    model_path = None
    vectorizer_path = None
    
    # Check in parent directory first (pages folder is subdirectory)
    parent_dir = os.path.dirname(BASE_DIR)
    if os.path.isfile(os.path.join(parent_dir, MODEL_FILENAME)) and \
       os.path.isfile(os.path.join(parent_dir, VECTORIZER_FILENAME)):
        model_path = os.path.join(parent_dir, MODEL_FILENAME)
        vectorizer_path = os.path.join(parent_dir, VECTORIZER_FILENAME)
    # Check in current directory
    elif os.path.isfile(os.path.join(BASE_DIR, MODEL_FILENAME)) and \
         os.path.isfile(os.path.join(BASE_DIR, VECTORIZER_FILENAME)):
        model_path = os.path.join(BASE_DIR, MODEL_FILENAME)
        vectorizer_path = os.path.join(BASE_DIR, VECTORIZER_FILENAME)
    
    if model_path and vectorizer_path:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        MODEL_LOADED = True
    else:
        st.error(f"⚠️ Model files not found in expected locations.")
        
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")

# --- Helper Functions ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

def style_sentiment(sentiment):
    """Applies color to the sentiment text for better visualization."""
    if sentiment == "Positive": # Teal
        return "background-color: #14B8A6; color: white"
    elif sentiment == "Negative": # Red
        return "background-color: #F43F5E; color: white"
    else: # Amber
        return "background-color: #F59E0B; color: black"

def run_bulk_analysis():
    """Defines the layout and logic for the Bulk Analysis page."""
    # Consistent styling with the main app
    st.markdown("""
        <style>
            .stButton > button {
                background: linear-gradient(90deg, #39FF14, #00b39f);
                color: white;
                border: none;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📂 Bulk Comment Analysis")
    st.markdown("Analyze multiple comments at once by pasting text.")

    # Define example bulk text
    EXAMPLE_BULK_TEXT = """Absolutely love this! It works perfectly and looks great.
Five stars! The team was incredibly supportive and quick to respond.
Best purchase I've made all year. Highly recommended!
The item was delivered on the expected date.
It functions as described in the product listing.
This is the third time I've ordered this product.
Completely disappointed. It stopped working after just one week.
The customer service was rude and unhelpful.
Way too expensive for what you get. A total rip-off.
I was expecting more, but the performance is just average."""

    # Check if model and vectorizer are loaded
    if not MODEL_LOADED or model is None or vectorizer is None:
        st.error("""
        🔴 **Error:** Failed to load the sentiment analysis model.
        
        Please ensure that:
        1. `sentiment_models.pkl` and `tfidf_vectorizer.pkl` exist in the project's root directory
        2. The files are not corrupted
        3. You have the necessary permissions to read these files
        """)
        
        # Show debug information
        with st.expander("🔍 Debug Information"):
            parent_dir = os.path.dirname(BASE_DIR)
            st.code(f"""
Current File Location: {BASE_DIR}
Parent Directory (Project Root): {parent_dir}

Checking Parent Directory:
- Model path: {os.path.join(parent_dir, MODEL_FILENAME)}
- Model exists: {os.path.exists(os.path.join(parent_dir, MODEL_FILENAME))}
- Vectorizer path: {os.path.join(parent_dir, VECTORIZER_FILENAME)}
- Vectorizer exists: {os.path.exists(os.path.join(parent_dir, VECTORIZER_FILENAME))}

Files in parent directory:
{os.listdir(parent_dir) if os.path.exists(parent_dir) else 'Directory not accessible'}

Checking Current Directory:
- Model path: {os.path.join(BASE_DIR, MODEL_FILENAME)}
- Model exists: {os.path.exists(os.path.join(BASE_DIR, MODEL_FILENAME))}

Files in current directory:
{os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'Directory not accessible'}
            """)
        
        st.stop()  # Stop execution if model isn't loaded

    # Initialize session state for bulk text area
    if 'bulk_user_input' not in st.session_state:
        st.session_state.bulk_user_input = ""

    st.button("Load Examples", on_click=lambda: st.session_state.update(bulk_user_input=EXAMPLE_BULK_TEXT.strip()))
    
    input_text = st.text_area(
        "Paste comments here, one per line:",
        height=250,
        placeholder="This service is amazing!\nThe product arrived damaged.\nIt works as expected.",
        key="bulk_user_input"
    )
    analyze_button = st.button("📊 Analyze Pasted Text", use_container_width=True)
    if analyze_button and input_text.strip():
        comments = [line.strip() for line in input_text.split('\n') if line.strip()]
    else:
        comments = []

    if comments:
        with st.spinner(f"Analyzing {len(comments)} comments..."):
            # Perform predictions
            X = vectorizer.transform(comments)
            predictions = model.predict(X)
            
            # Get confidence scores (match app.py logic)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
            else:
                # For LinearSVC and other models without predict_proba
                from utils import softmax
                decision_scores = model.decision_function(X)
                probabilities = softmax(decision_scores)

            confidence_scores = [prob[list(model.classes_).index(pred)] for pred, prob in zip(predictions, probabilities)]

            # Prepare data for display
            emoji_map = {"Positive": "😄", "Neutral": "😐", "Negative": "😞"}
            results_data = {
                "Comment": comments,
                "Sentiment": [p.capitalize() for p in predictions],
                "Confidence": confidence_scores,
                "Emoji": [emoji_map.get(p.capitalize(), "🤔") for p in predictions]
            }
            results_df = pd.DataFrame(results_data)
            
            # Add summary to history
            sentiment_counts = results_df['Sentiment'].value_counts()
            top_sentiment = sentiment_counts.idxmax()
            add_to_history(f"Bulk Analysis ({len(comments)} items)", top_sentiment)

            st.divider()
            st.subheader("📊 Analysis Summary")

            # Top Section: Chart and Download
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("<h6>Sentiment Distribution</h6>", unsafe_allow_html=True)
                st.bar_chart(sentiment_counts)

            with col2:
                st.write("") # Spacer
                st.write("")
                # Prepare CSV for download
                csv = convert_df_to_csv(results_df)
                st.download_button(
                   label="📥 Download Results as CSV",
                   data=csv,
                   file_name='bulk_sentiment_analysis.csv',
                   mime='text/csv',
                   use_container_width=True
                )
            
            # Word Cloud Section
            st.divider()
            st.subheader("☁️ Word Cloud")
            with st.spinner("Generating Word Cloud..."):
                fig = generate_wordcloud_plot(comments)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("Not enough text to generate a word cloud.")

            st.divider()

            # Bottom Section: Detailed Results
            st.markdown("<h6>Detailed Results</h6>", unsafe_allow_html=True)
            st.dataframe(
                results_df.style.applymap(style_sentiment, subset=['Sentiment'])
                                .format({"Confidence": "{:.2%}"})
                                .bar(subset=["Confidence"], color='#14B8A6', vmin=0, vmax=1),
                use_container_width=True,
                height=600
            )
    elif analyze_button and not input_text.strip():
        st.warning("⚠️ The text area is empty. Please paste some comments.")
    
    # Show history sidebar
    show_history_sidebar()

# Run the analysis
run_bulk_analysis()