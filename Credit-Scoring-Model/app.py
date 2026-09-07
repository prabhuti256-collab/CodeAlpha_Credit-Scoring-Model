import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("credit_scoring_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(
    page_title="Credit Scoring Model",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Scoring Model")
st.write("Predict an applicant's creditworthiness using a Machine Learning model.")

st.divider()

st.subheader("Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    checking = st.selectbox(
        "Checking Account",
        ["A11", "A12", "A13", "A14"]
    )

    duration = st.number_input(
        "Duration (months)",
        min_value=1,
        max_value=72,
        value=12
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=1,
        value=1000
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    job = st.selectbox(
        "Job",
        [0, 1, 2, 3]
    )

    housing = st.selectbox(
        "Housing",
        ["A151", "A152", "A153"]
    )

with col2:
    account_status = st.selectbox(
        "Credit History",
        ["A30", "A31", "A32", "A33", "A34"]
    )

    purpose = st.selectbox(
        "Purpose",
        ["A40", "A41", "A42", "A43", "A44", "A45",
         "A46", "A47", "A48", "A49", "A410"]
    )

    saving = st.selectbox(
        "Saving Account",
        ["A61", "A62", "A63", "A64", "A65"]
    )

    installment = st.number_input(
        "Installment Rate",
        min_value=1,
        max_value=4,
        value=2
    )

    personal_status = st.selectbox(
        "Personal Status",
        ["A91", "A92", "A93", "A94"]
    )

    existing_credits = st.number_input(
        "Existing Credits",
        min_value=1,
        max_value=4,
        value=1
    )

st.divider()

if st.button("🔮 Predict Credit Risk", use_container_width=True):

    # Remaining columns use common/default values
    data = {
        0: [checking],
        1: [duration],
        2: [account_status],
        3: [purpose],
        4: [credit_amount],
        5: [saving],
        6: ["A75"],
        7: [installment],
        8: ["A93"],
        9: ["A101"],
        10: [existing_credits],
        11: ["A121"],
        12: [age],
        13: ["A143"],
        14: ["A152"],
        15: [job],
        16: ["A173"],
        17: [1],
        18: ["A192"],
        19: ["A201"]
    }

    customer = pd.DataFrame(data)

    # Preprocess
    customer_processed = preprocessor.transform(customer)

    # Prediction
    prediction = model.predict(customer_processed)[0]
    probability = model.predict_proba(customer_processed)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("🟢 GOOD CREDIT")
    else:
        st.error("🔴 BAD CREDIT")

    st.write(
        f"**BAD Credit Probability:** {probability[0]:.2%}"
    )

    st.write(
        f"**GOOD Credit Probability:** {probability[1]:.2%}"
    )
    