# 📩 Spam Detection System

A complete NLP project that classifies SMS messages as **Spam** or **Ham** using **Natural Language Processing (NLP)** and **Machine Learning**.

---

# 🚀 Project Overview

This project builds a Spam Detection model using the SMS Spam Collection Dataset.

The project includes:

- Data Cleaning
- Text Preprocessing
- TF-IDF Feature Extraction
- Support Vector Machine (LinearSVC)
- Model Saving
- Prediction Script
- FastAPI REST API
- Interactive Swagger Documentation

---

# 📂 Project Structure

```text
Spam-Detection-System/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── api.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── test_preprocessing.py
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

Dataset:

**SMS Spam Collection Dataset**

Target Labels:

- Ham (0)
- Spam (1)

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn

---

# 🧹 NLP Pipeline

The preprocessing pipeline performs:

- Convert text to lowercase
- Remove URLs
- Remove numbers
- Remove punctuation
- Remove extra spaces
- Tokenization
- Stopword Removal
- Porter Stemming

---

# 🤖 Machine Learning Pipeline

Feature Extraction:

- TF-IDF Vectorizer

Classifier:

- Linear Support Vector Classifier (LinearSVC)

---

# 📈 Model Performance

Accuracy:

```text
97.97%
```

Classification Report

```text
Precision : 0.98
Recall    : 0.98
F1 Score  : 0.98
```

---

# 💾 Saved Models

The trained model is stored inside:

```text
models/
```

Files:

- svm_model.pkl
- tfidf_vectorizer.pkl

---

# 🌐 REST API

Run the API:

```bash
uvicorn api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 📥 Example Request

```json
{
    "text": "Congratulations! You won a free iPhone."
}
```

---

# 📤 Example Response

```json
{
    "message": "Congratulations! You won a free iPhone.",
    "prediction": "Spam"
}
```

---

# ▶️ How to Run

## 1. Clone Repository

```bash
git clone <repository-url>
```

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

## 3. Train Model

```bash
python src/train.py
```

## 4. Run Prediction

```bash
python src/predict.py
```

## 5. Start API

```bash
uvicorn api:app --reload
```

---

# 🎯 Future Improvements

- Add Lemmatization
- Hyperparameter Tuning
- Deploy using Docker
- Deploy on Render
- Build Web Interface
- Try Deep Learning Models
- Compare with BERT

---

# 👨‍💻 Author

**Mostafa Mohamed**

Faculty of Computers and Information

Artificial Intelligence Department
