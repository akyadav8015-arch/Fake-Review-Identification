import joblib
from preprocessing import clean_text

def predict_review(text):
    model = joblib.load("models/best_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    if hasattr(model, "predict_proba"):
        confidence = model.predict_proba(vector).max()
    else:
        confidence = "Not available for SVM"

    return prediction, confidence


if __name__ == "__main__":
    review = input("Enter review: ")
    pred, conf = predict_review(review)

    if pred == 1:
        print("Prediction: Genuine")
    else:
        print("Prediction: Fake")

    print("Confidence:", conf)
