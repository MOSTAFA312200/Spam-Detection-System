import joblib

from preprocessing import preprocess_text
# Load trained model
model = joblib.load("../models/svm_model.pkl")

# Load TF-IDF vectorizer
tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

def predict_message(text):

    # Preprocess text
    clean_text = preprocess_text(text)

    # Convert text to TF-IDF features
    vector = tfidf.transform([clean_text])

    # Predict
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "Spam"
    else:
        return "Ham"


if __name__ == "__main__":
    message = input("Enter your message: ")

    result = predict_message(message)

    print(f"Prediction: {result}")


