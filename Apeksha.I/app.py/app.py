import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
    page_title="Iris Predictor",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS to make background full-screen and main box compact
FLOWER_BG_URL = "https://images.unsplash.com/photo-1490750967868-88aa4486c946?q=80&w=2070&auto=format&fit=crop"

st.markdown(
    f"""
    <style>
    /* Full-screen flower background */
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url("{FLOWER_BG_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Make the calculation card compact and centered */
    .stMainBlockContainer {{
        max-width: 480px !important;
        background-color: rgba(255, 255, 255, 0.92);
        padding: 2rem !important;
        border-radius: 24px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        margin: 3rem auto !important;
    }}

    h2 {{
        color: #1b5e20;
        text-align: center;
        font-size: 1.8rem !important;
        margin-bottom: 0.2rem;
    }}
    
    .subtitle {{
        text-align: center;
        color: #555;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }}

    /* Compact button styling */
    .stButton>button {{
        background-color: #2e7d32;
        color: white !important;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        margin-top: 1rem;
    }}

    .stButton>button:hover {{
        background-color: #1b5e20;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Train machine learning model
@st.cache_resource
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    return clf, iris.target_names

model, target_names = train_model()

# Compact Screen Content
st.markdown("## 🌸 Iris Predictor")
st.markdown("<p class='subtitle'>Enter flower measurements below</p>", unsafe_allow_html=True)

# Compact 2-column input layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.8, step=0.1)
    sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.3, step=0.1)
    petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.3, step=0.1)

# Prediction Logic
if st.button("Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_features)[0]
    predicted_species = target_names[prediction].capitalize()

    st.success(f"**Species:** Iris {predicted_species}")