# 🕵️‍♂️ Fraud Detection with Machine Learning

This project implements a **credit card fraud detection system** using machine learning.  
It covers the full pipeline: data preprocessing, feature engineering, model training, evaluation, and exposing predictions via a **FastAPI REST API**.


**Note:** The dataset (`creditcard.csv`) is **not included** in this repository due to size and privacy constraints.  
You can download it from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it in the `data/raw/` directory.

---

## 📂 Project Structure

```bash
Fruad_Detection/
├── data/ # Raw
├── models/ # Saved trained models (joblib)
├── notebooks/ # EDA + experiments
├── plots/ # Plots od matrix and roc
├── reports/ # The report of evaluate.py
├── src/ # Source code
│ ├── data.py # Data loading & splitting
│ ├── features.py # Feature engineering
│ ├── model.py # Training & saving models
│ ├── evaluate.py # Evaluation metrics & plots
│ └── api.py # FastAPI app
├── test.py # Script for testing API predictions
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## 🚀 Features

- **Preprocessing & Feature Engineering**  
  Time-aware train/test split, feature scaling, and derived features.

- **Model Training**  
  Random Forest with cross-validation (configurable for other ML models).

- **Evaluation**  
  Confusion matrix, Precision-Recall curve, and detailed metrics.

- **API**  
  REST API built with **FastAPI** for real-time fraud probability predictions.

---

## ⚙️ Installation

Clone the repo and set up a virtual environment:

```bash
git clone https://github.com/KouroshBeheshtinejad/Fruad_Detection.git
cd Fruad_Detection

# Create and activate venv
python -m venv .venv
.venv\Scripts\activate   # On Windows
# source .venv/bin/activate   # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the Model

Run training pipeline to build and save the model:

```bash
python src/model.py
```

The trained model will be stored in `models/rf_model.joblib`.


### 2. Start the API Server

Launch FastAPI with Uvicorn:

```bash
uvicorn src.api:app --reload
```

API runs at:
👉 `http://127.0.0.1:8000`

Docs available at:
👉 `http://127.0.0.1:8000/docs`

### 3. Make Predictions
Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
-H "Content-Type: application/json" ^
-d "{\"Time\":0,\"V1\":-1.23,\"V2\":2.56,\"V3\":0.12, \"V4\":-0.45, \"V5\":0.67,
     \"V6\":-1.11,\"V7\":2.22,\"V8\":0.0,\"V9\":-0.89,\"V10\":1.01,
     \"V11\":0.56,\"V12\":-0.33,\"V13\":1.22,\"V14\":0.78,\"V15\":-0.77,
     \"V16\":0.99,\"V17\":-0.11,\"V18\":0.44,\"V19\":0.55,\"V20\":-0.66,
     \"V21\":0.77,\"V22\":-0.88,\"V23\":0.12,\"V24\":-0.45,\"V25\":0.22,
     \"V26\":-0.33,\"V27\":1.11,\"V28\":-0.22,\"Amount\":123.45}"
```
Or use `test.py` to send requests.

---

## 📊 Results

- Cross-validated Random Forest achieves high recall on fraud cases.
- Precision-Recall tradeoff analyzed with plots.
- Fraud probability returned as a float in API responses.

---

## 🛠️ Tech Stack

- Python 3.10+
- scikit-learn
- pandas, numpy
- matplotlib
- FastAPI + Uvicorn
- joblib

---

## 📝 Future Improvements

- Add deep learning models (e.g., LSTM/Autoencoder).
- Experiment with SMOTE/undersampling for imbalance.
- Deploy API to cloud (Heroku, AWS, etc.).

---

## 🤝 Contributing

Pull requests and discussions are welcome!
Please open an issue if you find a bug or want a feature.

## 📜 License

MIT License © 2025

## Author

**Kourosh Beheshtinejad**  
- Email: kouroshbnj@gmail.com  
- GitHub: [https://github.com/Kouroshbeheshtinejad](https://github.com/Kouroshbeheshtinejad)  