import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
import numpy as np

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    was_1d = False
    if x.ndim == 1:
        was_1d = True
        x = x.reshape(1, -1)
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    result = e_x / e_x.sum(axis=1, keepdims=True)
    return result.flatten() if was_1d else result

import streamlit as st
from datetime import datetime

def add_to_history(input_text, sentiment):
    """Adds an analysis result to the session history."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Store a summary if text is long
    summary = input_text[:50] + "..." if len(input_text) > 50 else input_text
    
    entry = {
        "text": summary,
        "sentiment": sentiment,
        "time": datetime.now().strftime("%H:%M:%S")
    }
    
    # Prepend to history (newest first)
    st.session_state.history.insert(0, entry)
    
    # Keep only last 10 entries
    if len(st.session_state.history) > 10:
        st.session_state.history.pop()