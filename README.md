# 🛡️ ChurnShield AI — Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-79.2%25-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **Predict customer churn before it happens — powered by Machine Learning**

ChurnShield AI is an end-to-end machine learning project that predicts whether a telecom customer is likely to churn (cancel their subscription). It includes a full ML pipeline (`churn.py`) and a polished Streamlit web app (`app.py`) for real-time predictions.

---

## 📸 App Preview

The app features a sleek dark-themed UI with real-time churn prediction, probability scores, and actionable business recommendations.

---

## 📁 Project Structure

```
Ai-ml-project/
│
├── churn.py                          # ML pipeline: train, tune, evaluate, save model
├── app.py                            # Streamlit web app for predictions
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (IBM Telco)
│
├── churn_model.pkl                   # Trained Random Forest model
├── scaler.pkl                        # StandardScaler for feature normalization
├── feature_columns.pkl               # Exact column order used during training
│
├── confusion_matrix.png              # Model evaluation — confusion matrix
├── feature_importance.png            # Top 10 churn-driving features
│
└── README.md
```

---

## 🧠 ML Pipeline Overview (`churn.py`)

| Step | Description |
|------|-------------|
| 1 | Load dataset (7,043 customer records) |
| 2 | Clean data — handle missing values, encode target |
| 3 | Feature engineering — 4 new features added |
| 4 | One-hot encode categorical variables |
| 5 | Train/test split (80/20, stratified) |
| 6 | Scale features with `StandardScaler` |
| 7 | Train 3 models: Random Forest, Gradient Boosting, Logistic Regression |
| 8 | Hyperparameter tuning via `GridSearchCV` (5-fold CV) |
| 9 | Evaluate best model — accuracy, recall, classification report |
| 10 | Save model, scaler, and feature column order |

### 🔧 Engineered Features

```python
ChargesPerMonth  = TotalCharges / (tenure + 1)
HighSpender      = 1 if MonthlyCharges > 70 else 0
LongTermCustomer = 1 if tenure > 24 else 0
NewCustomer      = 1 if tenure < 6 else 0
```

---

## 📊 Model Performance

| Model | Accuracy | Churn Recall |
|-------|----------|-------------|
| Random Forest | **79.2%** | — |
| Gradient Boosting | — | — |
| Logistic Regression | — | — |

> 🏆 **Best Model:** Random Forest (after hyperparameter tuning with GridSearchCV)  
> **Algorithm:** `RandomForestClassifier` with `class_weight="balanced"`  
> **Tuned on:** `n_estimators`, `max_depth`, `min_samples_split`

---

## 🖥️ Web App Features (`app.py`)

Built with **Streamlit**, the ChurnShield AI app allows users to:

- Input customer profile details (demographics, services, billing)
- Get an instant **churn probability score**
- See a visual **probability bar**
- Receive **business action recommendations** based on the prediction

### Input Categories

- **Customer Profile** — Gender, Senior Citizen, Partner, Dependents, Tenure
- **Internet & Services** — Internet type, Online Security, Backup, Streaming, etc.
- **Billing & Contract** — Contract type, Payment method, Monthly/Total charges

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Rohanindia/Ai-ml-project.git
cd Ai-ml-project
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 3. Train the Model

```bash
python churn.py
```

This will generate `churn_model.pkl`, `scaler.pkl`, and `feature_columns.pkl`.

### 4. Launch the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

---

## 📂 Dataset

**IBM Telco Customer Churn Dataset**  
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Records: 7,043 customers  
- Features: 21 original + 4 engineered = **34 total features**  
- Target: `Churn` (Yes / No)

---

## 👤 Author

**Rohan**  
GitHub: [@Rohanindia](https://github.com/Rohanindia)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
