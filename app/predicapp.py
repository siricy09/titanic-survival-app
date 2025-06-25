import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from scipy.special import expit

# Load model and training data
model = joblib.load('titanic_model.pkl')
X_train = joblib.load('X_train.pkl')

st.title("Titanic Survival Prediction App")
st.markdown("Enter the passenger details below to predict whether they would have survived the Titanic disaster.")

# User inputs
pclass = st.selectbox("Ticket Class (1 = Upper, 2 = Middle, 3 = Lower)", [1, 2, 3])
sex = st.radio("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare Paid", 0, 500, 50)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
embarked = st.radio("Port of Embarkation", ['Cherbourg', 'Queenstown', 'Southampton'])

# Preprocess inputs
sex_encoded = 0 if sex == 'male' else 1
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

input_data = np.array([[pclass, sex_encoded, age, fare, sibsp, parch, embarked_Q, embarked_S]])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success("This passenger would have SURVIVED.")
    else:
        st.error("This passenger would NOT have survived.")

    # Text progress bar
    percent = int(proba[1] * 100)
    blocks = int(percent / 2)
    bar = '█' * blocks + '░' * (50 - blocks)
    st.subheader("Survival Chance")
    st.text(f"[{bar}] {percent}%")

    # SHAP Explanation
    st.subheader("Feature Impact Explanation")
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots()
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_data[0],
        feature_names=['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S']
    ), max_display=8, show=False)
    st.pyplot(fig)

    # SHAP Probability Calculation
    log_odds = explainer.expected_value + shap_values[0].sum()
    probability_from_shap = expit(log_odds)
    percent_from_shap = round(probability_from_shap * 100, 2)
    
    st.caption("Model: Logistic Regression | Trained on Titanic dataset")
    st.markdown(f"**SHAP Predicted Survival Probability:** {percent_from_shap}%")
    st.info("Positive (red) values push towards survival, negative (blue) push towards not surviving.")

