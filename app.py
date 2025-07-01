import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('random_forest_1gram_model.pkl')
vectorizer = joblib.load('tfidf_1gram_vectorizer.pkl')

# -------- Custom CSS (Web-Style Design) --------
st.markdown("""
    <style>
    /* Full-page gradient background */
    .stApp {
        background: linear-gradient(to right, #1f4037, #99f2c8);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title & Name Styles */
    .title-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 30px;
        background-color: rgba(0,0,0,0.3);
        color: white;
    }

    .title-bar h1 {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }

    .title-bar h3 {
        font-size: 18px;
        font-weight: 500;
        margin: 0;
        color: #ffe680;
        font-style: italic;
    }

    /* Glassmorphism card */
    .glass-box {
        margin: auto;
        width: 60%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 30px;
        margin-top: 50px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(12px);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .glass-box h2 {
        color: #00ffe7;
        margin-bottom: 20px;
    }

    /* Styled TextArea */
   .stTextArea textarea {
    background-color: #ffffff;
    color: #000000;
    font-size: 16px;
    border-radius: 8px;
    padding: 10px;
    border: 1px solid #ccc;
}

    /* Submit Button */
    .stButton > button {
        background-color: #00cc99;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }

    /* Result styling */
    .fake-box {
        background-color: #ff4d4d;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 20px;
    }

    .true-box {
        background-color: #33cc66;
        padding: 15px;
        border-radius: 10px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -------- Top Header --------
st.markdown("""
<div class="title-bar">
    <h1>Fake News Detection</h1>
    <h3>Hassan Bahaar</h3>
</div>
""", unsafe_allow_html=True)

# -------- Form Box --------
st.markdown('<div class="glass-box">', unsafe_allow_html=True)
st.markdown("<h2>📰 Check Your News Article</h2>", unsafe_allow_html=True)

user_input = st.text_area("Paste your news article or headline:")

if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter a news article to check.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        if prediction[0] == 0:
            st.markdown('<div class="fake-box">🚨 This news is likely FAKE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="true-box">✅ This news is likely TRUE</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
