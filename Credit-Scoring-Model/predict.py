import joblib
import pandas as pd

# Load trained model and preprocessor
model = joblib.load("credit_scoring_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

print("=" * 50)
print("       CREDIT SCORING PREDICTION")
print("=" * 50)

# Example customer data
data = {
    0: ["A11"],
    1: [6],
    2: ["A34"],
    3: ["A43"],
    4: [1169],
    5: ["A65"],
    6: ["A75"],
    7: [4],
    8: ["A93"],
    9: ["A101"],
    10: [4],
    11: ["A121"],
    12: [67],
    13: ["A143"],
    14: ["A152"],
    15: [2],
    16: ["A173"],
    17: [1],
    18: ["A192"],
    19: ["A201"]
}

# Convert to DataFrame
customer = pd.DataFrame(data)

# Preprocess customer data
customer_processed = preprocessor.transform(customer)

# Make prediction
prediction = model.predict(customer_processed)[0]

# Probability
probability = model.predict_proba(customer_processed)[0]

print("\nPrediction Result:")
print("-" * 30)

if prediction == 1:
    print("Credit Risk: GOOD")
else:
    print("Credit Risk: BAD")

print(f"Probability of BAD credit: {probability[0]:.2%}")
print(f"Probability of GOOD credit: {probability[1]:.2%}")

print("=" * 50)
