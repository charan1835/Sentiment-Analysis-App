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
    if sentiment == "Positive":
        return "background-color: #28a745; color: white" # Green
    elif sentiment == "Negative":
        return "background-color: #dc3545; color: white" # Red
    else:
        return "background-color: #ffc107; color: black" # Yellow/Amber

def run_bulk_analysis():
    """Defines the layout and logic for the Bulk Analysis page."""
    # Consistent styling with the main app
    st.markdown("""
        <style>
            .stButton button {
                background: linear-gradient(90deg, #39FF14, #00b39f);
                color: white;
                border: none;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📂 Bulk Comment Analysis")
    st.markdown("Analyze multiple comments at once by pasting text or uploading a file.")

    # Load the model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    if model is None or vectorizer is None:
        st.error("🔴 **Error:** Model or vectorizer files not found. Please ensure they are in the root directory.")
        return

    # --- Input Tabs ---
    tab1, tab2 = st.tabs(["✍️ Paste Text", "📤 Upload File"])

    with tab1:
        input_text = st.text_area(
            "Paste comments here, one per line:",
            height=250,
            placeholder="This service is amazing!\nThe product arrived damaged.\nIt works as expected."
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
                decision_scores = model.decision_function(X)
                probabilities = np.array([_softmax(scores) for scores in decision_scores])
            
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
                                    .bar(subset=["Confidence"], color='#39FF14', vmin=0, vmax=1),
                    use_container_width=True,
                    height=400
                )
    elif analyze_button and not input_text.strip():
        st.warning("⚠️ The text area is empty. Please paste some comments.")

# This check ensures the code runs only when this script is executed directly
# or by Streamlit's page navigation.
if __name__ == "__main__":
    run_bulk_analysis()