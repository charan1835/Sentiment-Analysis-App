import streamlit as st
import pandas as pd
import os
import joblib
from view_utils import main_page_styles, generate_wordcloud_plot, show_history_sidebar
import matplotlib.pyplot as plt
from utils import add_to_history

# --- Project Root Directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Constants ---
MODEL_FILENAME = "sentiment_models.pkl"
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"

# Define model and vectorizer at module level
model = None
vectorizer = None
MODEL_LOADED = False

# --- Load Model and Vectorizer ---
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

def run_file_upload_analysis():
    """Defines the layout and logic for the File Upload Analysis page."""
    # Apply shared styles
    main_page_styles()

    st.title("📤 File Upload Analysis")
    st.markdown("Upload a `.txt` or `.csv` file containing comments to analyze them in bulk.")

    # Check if model and vectorizer are loaded
    if not MODEL_LOADED or model is None or vectorizer is None:
        st.error("🔴 **Error:** Failed to load the sentiment analysis model.")
        st.stop()

    # File Uploader
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])

    if uploaded_file is not None:
        comments = []
        try:
            if uploaded_file.name.endswith('.txt'):
                # Read text file line by line
                stringio = uploaded_file.getvalue().decode("utf-8")
                comments = [line.strip() for line in stringio.split('\n') if line.strip()]
            elif uploaded_file.name.endswith('.csv'):
                # Read CSV file
                df = pd.read_csv(uploaded_file)
                # Assume the first column contains the text if not specified
                # Ideally, we'd ask the user to select the column, but for simplicity we'll take the first object column or the first column
                possible_cols = df.select_dtypes(include=['object']).columns
                if len(possible_cols) > 0:
                    target_col = possible_cols[0]
                    comments = df[target_col].dropna().astype(str).tolist()
                else:
                    comments = df.iloc[:, 0].dropna().astype(str).tolist()
            
            if not comments:
                st.warning("⚠️ No valid text found in the uploaded file.")
                return

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
                
                # Add summary to history
                sentiment_counts = results_df['Sentiment'].value_counts()
                top_sentiment = sentiment_counts.idxmax()
                add_to_history(f"File Upload ({len(comments)} items)", top_sentiment)

                st.divider()
                st.subheader("📊 Analysis Summary")

                # Top Section: Pie Chart and Download
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("<h6>Sentiment Distribution</h6>", unsafe_allow_html=True)
                    
                    # Create Pie Chart using Matplotlib
                    fig, ax = plt.subplots(figsize=(6, 6))
                    # Define colors matching the app theme
                    colors = {'Positive': '#14B8A6', 'Negative': '#F43F5E', 'Neutral': '#F59E0B'}
                    chart_colors = [colors.get(label, '#9CA3AF') for label in sentiment_counts.index]
                    
                    wedges, texts, autotexts = ax.pie(
                        sentiment_counts, 
                        labels=sentiment_counts.index, 
                        autopct='%1.1f%%', 
                        startangle=90,
                        colors=chart_colors,
                        textprops=dict(color="white")
                    )
                    
                    # Make the background transparent
                    fig.patch.set_alpha(0)
                    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                    
                    st.pyplot(fig)

                with col2:
                    st.write("") # Spacer
                    st.write("")
                    # Prepare CSV for download
                    csv = convert_df_to_csv(results_df)
                    st.download_button(
                       label="📥 Download Results as CSV",
                       data=csv,
                       file_name='file_upload_sentiment_analysis.csv',
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

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
    # Show history sidebar
    show_history_sidebar()

# Run the analysis
run_file_upload_analysis()
