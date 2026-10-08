# 💳 Credit Scoring Model

A machine learning project that predicts the **creditworthiness of individuals** using financial and demographic information such as income, debt, payment history, and other customer attributes.

This project was developed as part of a **CodeAlpha Machine Learning Internship** and includes multiple classification algorithms, model evaluation, and an interactive Streamlit application.

---

## 🚀 Project Overview

Credit scoring is an important application of machine learning in the financial sector. A credit scoring model helps estimate whether an individual is likely to be considered a **good or bad credit risk** based on historical financial information.

This project performs:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Classification model training
* Model comparison
* Performance evaluation
* Creditworthiness prediction
* Interactive Streamlit deployment

---

## 🎯 Objective

The primary objective is to build a machine learning model that classifies applicants based on their creditworthiness.

The model predicts two classes:

| Prediction  | Meaning                                    |
| ----------- | ------------------------------------------ |
| 🟢 **GOOD** | Applicant is considered creditworthy       |
| 🔴 **BAD**  | Applicant may represent higher credit risk |

---

## 🤖 Machine Learning Models

The project compares multiple classification algorithms:

### 1. Logistic Regression

A linear classification algorithm used as a baseline model for predicting creditworthiness.

### 2. Decision Tree

A tree-based model that makes predictions using a series of decision rules.

### 3. Random Forest

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance and robustness.

The models are evaluated and the best-performing model is selected for prediction.

---

## 📂 Dataset

The dataset contains financial and demographic information about credit applicants.

Important features include:

* Age
* Income
* Debt
* Payment history
* Employment information
* Credit-related attributes
* Purpose of credit

### Target Variable

```text
Creditworthiness
```

The target is classified into:

```text
GOOD
BAD
```

---

## 🔄 Project Workflow

```text
Credit Dataset
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Selection
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Model Comparison
      ↓
Model Evaluation
      ↓
Best Model Selection
      ↓
Credit Prediction
      ↓
Streamlit Application
```

---

## 📊 Model Evaluation

The models are evaluated using several classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1 Score**
* **ROC-AUC**
* **Confusion Matrix**

Using multiple metrics provides a better understanding of how well the model performs on both GOOD and BAD credit classes.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Streamlit**
* **Machine Learning**

---

## 📁 Project Structure

```text
CodeAlpha_Credit-Scoring-Model/
│
├── dataset/
│   └── credit_data.csv
│
├── models/
│   └── credit_model.pkl
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/prabhuti256-collab/CodeAlpha_Credit-Scoring-Model.git
```

### 2. Navigate to the Project

```bash
cd CodeAlpha_Credit-Scoring-Model
```

### 3. Create a Virtual Environment

```bash
py -3.12 -m venv venv
```

### 4. Activate the Virtual Environment

```powershell
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Model

Run:

```bash
python train_model.py
```

The training script:

1. Loads the credit dataset
2. Cleans and preprocesses the data
3. Separates features and target
4. Splits the data into training and testing sets
5. Trains multiple classification models
6. Evaluates model performance
7. Saves the selected model

---

## 🔮 Make Predictions

Run:

```bash
python predict.py
```

The prediction script loads the trained model and predicts the creditworthiness of a new applicant.

Example:

```text
========================================
      CREDIT SCORING PREDICTION
========================================

Prediction: GOOD
Probability of GOOD Credit: 96.23%
Probability of BAD Credit: 3.77%

========================================
```

---

## 🌐 Streamlit Application

Run the interactive application:

```bash
streamlit run app.py
```

The Streamlit application allows users to enter applicant information and receive a creditworthiness prediction.

### Example

```text
Credit Score Prediction

Prediction: 🟢 GOOD

GOOD Probability: 96.23%
BAD Probability: 3.77%
```

---

## 💡 Key Features

✅ Creditworthiness classification
✅ Multiple machine learning algorithms
✅ Logistic Regression
✅ Decision Tree
✅ Random Forest
✅ Model performance comparison
✅ Classification metrics
✅ Probability-based predictions
✅ Interactive Streamlit application
✅ Saved trained model
✅ Internship project suitable for portfolio

---

## 🔐 Responsible Use

Credit scoring models can influence important financial decisions. A machine learning prediction should therefore be treated as **decision support rather than a definitive judgment**.

Real-world credit systems should include:

* Fairness and bias testing
* Explainable predictions
* Regulatory compliance
* Secure handling of financial data
* Human review where appropriate

---

## 🔮 Future Improvements

* Hyperparameter tuning
* Cross-validation
* XGBoost and LightGBM models
* Class imbalance handling
* SHAP-based explainable AI
* Feature importance visualization
* Model monitoring
* Fairness and bias analysis
* REST API deployment
* Cloud deployment

---

## 🏢 Internship

This project was developed as part of the **CodeAlpha Machine Learning Internship**.

The project demonstrates practical experience in:

* Machine Learning
* Classification
* Data Preprocessing
* Model Evaluation
* Python
* Streamlit
* GitHub Project Development

---

## 👩‍💻 Author

**Prabhuti**

B.Tech — Artificial Intelligence & Machine Learning

GitHub:
https://github.com/prabhuti256-collab

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
