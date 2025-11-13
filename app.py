import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# =====================================
# DARK / LIGHT MODE TOGGLE (v1.38.0+)
# =====================================
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

# Toggle buttons — top-right corner
col1, col2 = st.columns([10, 1])
with col2:
    if st.session_state.theme == 'light':
        st.button('Dark', on_click=toggle_theme, use_container_width=True)
    else:
        st.button('Light', on_click=toggle_theme, use_container_width=True)

# Apply theme with full styling
if st.session_state.theme == 'dark':
    st.markdown("""
    <style>
    .stApp {
        background: #0e1117;
        color: #e0e0e0;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1f2937 !important;
        color: #e0e0e0 !important;
        border: 1px solid #374151 !important;
    }
    .stButton > button {
        background-color: #374151;
        color: #e0e0e0;
    }
    .stMarkdown, .stCaption, .stSubheader, h1, h2, h3 {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1c23;
    }
    .stAlert {
        background-color: #1f2937;
        border: 1px solid #374151;
    }
    </style>
    """, unsafe_allow_html=True)
