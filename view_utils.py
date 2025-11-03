import streamlit as st

def main_page_styles():
    """Injects custom CSS for the app for a consistent look and feel."""
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
        </style>
    """, unsafe_allow_html=True)