# 💳 Credit Scoring Model

## 📌 Project Overview

This project predicts an individual's creditworthiness using historical financial information and Machine Learning classification algorithms.

The project compares **Logistic Regression, Decision Tree, and Random Forest** models and evaluates them using Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

## 🎯 Objective

The main objective is to develop a Machine Learning model that can classify applicants based on their credit risk and help identify whether an applicant is likely to have **Good Credit** or **Bad Credit**.

## 📊 Dataset

* Number of records: **1,000**
* Number of features: **20**
* Target variable: **Credit Risk**
* Good Credit: **700**
* Bad Credit: **300**

The dataset contains financial and personal information such as credit amount, loan duration, age, account information, savings, housing, and other credit-related attributes.

## 🧹 Data Preprocessing

The following preprocessing techniques were applied:

* Removed unnecessary index information
* Separated features and target variable
* Identified numerical and categorical features
* Handled categorical variables using encoding
* Applied preprocessing before model training
* Split the dataset into training and testing sets

## 🤖 Machine Learning Models

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## 📈 Model Performance

| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **78.00%** | **66.67%** | **53.33%** | **59.26%** | **80.40%** |
| Decision Tree           |     61.50% |     39.51% |     53.33% |     45.39% |     65.07% |
| Random Forest           |     76.00% |     65.79% |     41.67% |     51.02% |     78.38% |

## 🏆 Best Model

**Logistic Regression** achieved the best overall performance.

* Accuracy: **78%**
* Precision: **66.67%**
* Recall: **53.33%**
* F1 Score: **59.26%**
* ROC-AUC: **80.40%**

Therefore, Logistic Regression was selected as the final model.

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application.

Users can enter applicant information and receive a credit-risk prediction along with the predicted probabilities.

### Run the application

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

## 📁 Project Structure

```text
Credit-Scoring-Model/
│
├── dataset/
│   └── credit_data.csv
│
├── credit_scoring_model.pkl
├── preprocessor.pkl
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Train the model

```bash
python train_model.py
```

### Make a prediction

```bash
python predict.py
```

### Launch the web application

```bash
python -m streamlit run app.py
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Machine Learning

## 🔮 Future Improvements

* Hyperparameter tuning
* Cross-validation
* Improved handling of class imbalance
* Feature engineering
* Explainable AI for credit decisions
* More advanced classification models
* Cloud deployment
* Improved user interface

## ⚠️ Disclaimer

This project is created for educational and demonstration purposes. Its predictions should not be used as the sole basis for real-world financial or lending decisions.

## 👩‍💻 Author

**Prabhuti**

BTech – Artificial Intelligence & Machine Learning

