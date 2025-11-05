import streamlit as st
import pandas as pd
import os
import sys


# Add the parent directory to the path to import the loading function
# This allows the page to find the `app.py` module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the cached model loading function from the main app
from app import load_model_and_vectorizer

# --- Helper Functions ---
@st.cache_data
def process_bulk_text(comments, model, vectorizer):
    """Processes a list of comments and returns a results DataFrame."""

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
    st.markdown("Analyze multiple comments at once by pasting text or uploading a file.")

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

    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    if model is None or vectorizer is None:
        st.error("🔴 **Error:** Model or vectorizer files not found. Please make sure `sentiment_models.pkl` and `tfidf_vectorizer.pkl` are in the project's root directory.")
        return

    # Initialize session state for bulk text area
    if 'bulk_user_input' not in st.session_state:
        st.session_state.bulk_user_input = ""

    # --- Input Tabs ---
    tab1, tab2 = st.tabs(["✍️ Paste Text", "📤 Upload File"])

    with tab1:
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

    with tab2:
        st.info("File upload functionality is coming soon!", icon="⏳")
        # Placeholder for future file uploader
        st.file_uploader("Upload a .txt or .csv file", type=['txt', 'csv'], disabled=True)

    if comments:
        with st.spinner(f"Analyzing {len(comments)} comments..."):
            # Perform predictions
            X = vectorizer.transform(comments)
            predictions = model.predict(X)
            
            # Get confidence scores
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
            else:
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

            st.divider()
            st.subheader("📊 Analysis Summary")

            col1, col2 = st.columns([1.5, 2.5])

            with col1:
                st.markdown("<h6>Sentiment Distribution</h6>", unsafe_allow_html=True)
                sentiment_counts = results_df['Sentiment'].value_counts()
                st.bar_chart(sentiment_counts)

                # Prepare CSV for download
                csv = convert_df_to_csv(results_df)
                st.download_button(
                   label="📥 Download Results as CSV",
                   data=csv,
                   file_name='bulk_sentiment_analysis.csv',
                   mime='text/csv',
                   use_container_width=True
                )

            with col2:
                st.markdown("<h6>Detailed Results</h6>", unsafe_allow_html=True)
                st.dataframe(
                    results_df.style.applymap(style_sentiment, subset=['Sentiment'])
                                    .format({"Confidence": "{:.2%}"})
                                    .bar(subset=["Confidence"], color='#14B8A6', vmin=0, vmax=1),
                    use_container_width=True,
                    height=400
                )
    elif analyze_button and not input_text.strip():
        st.warning("⚠️ The text area is empty. Please paste some comments.")

# This check ensures the code runs only when this script is executed directly
# or by Streamlit's page navigation.
if __name__ == "__main__":
    run_bulk_analysis()