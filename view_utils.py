import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def main_page_styles():
    """Injects custom CSS for the app for a consistent look and feel."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

            /* --- Base & Fonts (High-Contrast Dark Theme) --- */
            body {
                background-color: #111827; /* Dark Charcoal */
                color: #F9FAFB; /* Off-White */
                font-family: 'Poppins', sans-serif;
            }

            /* --- Main Container --- */
            .main {
                background: #1F2937; /* Lighter Charcoal */
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid #374151; /* Subtle Border */
                box-shadow: 0px 8px 30px rgba(0,0,0,0.2);
            }

            /* --- Headers & Text --- */
            h1 {
                background: -webkit-linear-gradient(45deg, #2DD4BF, #38BDF8); /* Teal to Cyan Gradient */
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5rem !important;
                font-weight: 700;
            }

            /* --- Widgets (Text Area & Buttons) --- */
            .stTextArea textarea {
                background-color: #111827 !important;
                color: #F9FAFB !important;
                border-radius: 10px;
                border: 1px solid #374151;
            }
            .stButton button {
                background: linear-gradient(90deg, #14B8A6, #0EA5E9); /* Teal to Blue Gradient */
                color: white;
                border-radius: 10px;
                padding: 0.6rem 1.5rem;
                font-weight: 600;
                border: none;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                transform: scale(1.05);
                box-shadow: 0px 4px 25px rgba(20, 184, 166, 0.4); /* Teal Glow */
            }
        </style>
    """, unsafe_allow_html=True)

def generate_wordcloud_plot(text_data):
    """Generates a word cloud plot from a list of text."""
    if not text_data:
        return None
        
    text = " ".join(text_data)
    # Use a font compatible with the dark theme or just standard
    wordcloud = WordCloud(width=800, height=400, background_color='#1F2937', colormap='viridis').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # Set figure background to match app theme
    fig.patch.set_facecolor('#1F2937')
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

def show_history_sidebar():
    """Displays the analysis history in the sidebar."""
    with st.sidebar:
        st.header("🕒 History")
        if 'history' not in st.session_state or not st.session_state.history:
            st.info("No analysis history yet.")
        else:
            for item in st.session_state.history:
                emoji = "😐"
                if item['sentiment'] == "Positive":
                    emoji = "😄"
                elif item['sentiment'] == "Negative":
                    emoji = "😞"
                
                st.markdown(f"**{item['time']}** {emoji} {item['sentiment']}")
                st.caption(f"{item['text']}")
                st.divider()