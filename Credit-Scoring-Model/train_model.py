import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load German Credit dataset
data = pd.read_csv(
    "dataset/german.data",
    sep=" ",
    header=None
)

print("Dataset shape:", data.shape)
print(data.head())

# Target is the last column
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert target:
# 1 = Good credit
# 2 = Bad credit
y = y.map({1: 0, 2: 1})

print("\nTarget distribution:")
print(y.value_counts())

# Treat all columns as categorical/numerical based on data
categorical_columns = X.select_dtypes(include=["str"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical_columns
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        )
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=6
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

# Compare models
results_df = pd.DataFrame(results).T

print("\n\nMODEL COMPARISON")
print(results_df)

# Find best model based on F1 score
best_model = results_df["F1 Score"].idxmax()

print("\nBest Model:", best_model)

import joblib

# Get the trained Logistic Regression model
best_model = models["Logistic Regression"]

# Save the model
joblib.dump(best_model, "credit_scoring_model.pkl")

# Save the preprocessor
joblib.dump(preprocessor, "preprocessor.pkl")

print("\nBest model saved successfully!")
print("Model: credit_scoring_model.pkl")
print("Preprocessor: preprocessor.pkl")
