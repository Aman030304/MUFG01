import streamlit as st
import pandas as pd
import pickle
import os

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

st.title('Heart Disease Prediction')

st.write("""
Enter the patient's information to predict the likelihood of heart disease.
""")

# Input fields including all required features
age = st.slider('Age', 18, 100, 50)
sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
chest_pain_type = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
resting_blood_pressure = st.slider('Resting Blood Pressure', 80, 200, 120)
cholesterol = st.slider('Cholesterol', 100, 600, 200)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
resting_ecg = st.selectbox('Resting ECG', [0, 1, 2])
max_heart_rate = st.slider('Maximum Heart Rate Achieved', 60, 220, 150)
exercise_induced_angina = st.selectbox('Exercise Induced Angina', [0, 1])
st_depression = st.slider('ST Depression', 0.0, 6.2, 1.0)
st_slope = st.selectbox('ST Slope', [0, 1, 2])  # Added
num_major_vessels = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])  # Added
thalassemia = st.selectbox('Thalassemia', [0, 1, 2, 3])  # Added

# Create DataFrame with all required features
input_data = pd.DataFrame([[
    age, sex, chest_pain_type, resting_blood_pressure, cholesterol,
    fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_induced_angina,
    st_depression, st_slope, num_major_vessels, thalassemia
]], columns=[
    'age', 'sex', 'chest_pain_type', 'resting_blood_pressure',
    'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
    'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels',
    'thalassemia'
])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.write(f'Probability of Heart Disease: {probability:.2%}')
    
    if prediction[0] == 1:
        st.error('Prediction: High likelihood of Heart Disease')
    else:
        st.success('Prediction: Low likelihood of Heart Disease')