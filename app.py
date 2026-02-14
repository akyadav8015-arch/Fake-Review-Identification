import streamlit as st
import joblib
import os

from src.preprocessing import clean_text
from src.database import init_db, insert_log, fetch_logs
from src.bert_predict import predict_bert

# Initialize database
init_db()

# Page config
st.set_page_config(
    page_title="Fake Review Detection System",
    layout="wide"
)

st.title("🛡 Fake Product Review Identification System")
st.markdown("AI-powered detection using ML and DistilBERT")

# Load ML model
ml_model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Sidebar
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose Model",
    ["Traditional ML", "DistilBERT"]
)

# Input area
review_text = st.text_area("Enter Product Review")

if st.button("Predict"):

    if review_text.strip() == "":
        st.warning("Please enter review text.")
    else:

        if model_choice == "Traditional ML":
            cleaned = clean_text(review_text)
            vector = vectorizer.transform([cleaned])

            prediction = ml_model.predict(vector)[0]

            if hasattr(ml_model, "predict_proba"):
                confidence = ml_model.predict_proba(vector).max()
            else:
                confidence = 0.85  # fallback for SVM

            label = "Genuine" if prediction == 1 else "Fake"

        else:
            label, confidence = predict_bert(review_text)

        # Display result
        if label == "Genuine":
            st.success(f"Prediction: {label}")
        else:
            st.error(f"Prediction: {label}")

        st.info(f"Confidence Score: {round(confidence * 100, 2)}%")

        # Save to database
        insert_log(review_text, label, float(confidence), model_choice)

# Display Logs
st.markdown("## 📊 Prediction History")

logs = fetch_logs()

if logs:
    for log in logs:
        st.write(f"ID: {log[0]}")
        st.write(f"Review: {log[1][:150]}...")
        st.write(f"Prediction: {log[2]}")
        st.write(f"Confidence: {round(log[3]*100,2)}%")
        st.write(f"Model Used: {log[4]}")
        st.write(f"Timestamp: {log[5]}")
        st.markdown("---")
else:
    st.write("No predictions yet.")
