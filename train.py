import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import preprocess_text

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")
print(df)
print(df.columns)

# Keep only required columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert labels to numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


df["message"] = df["message"].apply(preprocess_text)

print(df.head())



# Create TF-IDF Vectorizer
tfidf = TfidfVectorizer()

# Transform text into numerical features
X = tfidf.fit_transform(df["message"])

# Target labels
y = df["label"]

print(X.shape)
print(y.shape)


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())


# Train model
model = LinearSVC(C=10, random_state=42)

model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

# Classification Report
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)


# Save trained model
joblib.dump(
    model,
    "models/svm_model.pkl"
)

# Save TF-IDF vectorizer
joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)

print("Model and TF-IDF saved successfully!")