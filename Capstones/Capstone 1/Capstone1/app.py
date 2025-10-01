import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import preprocess_input


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


@st.cache_resource
def load_artifacts():
    # Prefer explicitly saved best model; fallback to lasso/others
    best_candidates = list(MODELS_DIR.glob("best_*_regression.pkl"))
    model_path = best_candidates[0] if best_candidates else None
    if model_path is None:
        # Try known filenames
        for name in ["lasso_regression.pkl", "ridge_regression.pkl", "linear_regression.pkl"]:
            path = MODELS_DIR / name
            if path.exists():
                model_path = path
                break
    if model_path is None:
        raise FileNotFoundError("No trained model found in models/ directory.")

    model = joblib.load(model_path)

    scaler_path = MODELS_DIR / "scaler.pkl"
    encoder_path = MODELS_DIR / "encoder.pkl"
    if not scaler_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Missing scaler.pkl or encoder.pkl in models/. Run training (python main.py) first.")
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    return model, scaler, encoder, model_path.name


def sidebar_info():
    st.sidebar.title("Project Overview")
    st.sidebar.markdown(
        """
        **Objective**: Predict manufacturing equipment output (Parts_Per_Hour) using linear models (Linear, Ridge, Lasso).

        **Dataset**: 1000 samples, 19 features with operational, environmental, and equipment metrics.

        **Business Value**:
        - Predictive maintenance
        - Production planning
        - Cost optimization and OEE improvement
        """
    )


def build_input_form():
    st.header("Input Features")
    with st.form("prediction_form"):
        # Numerical inputs (use reasonable ranges; adjust as needed)
        injection_temperature = st.slider("Injection_Temperature", 150.0, 260.0, 210.0, 0.1)
        injection_pressure = st.slider("Injection_Pressure", 80.0, 160.0, 120.0, 0.1)
        cycle_time = st.slider("Cycle_Time", 15.0, 50.0, 30.0, 0.1)
        cooling_time = st.slider("Cooling_Time", 5.0, 20.0, 12.0, 0.1)
        material_viscosity = st.slider("Material_Viscosity", 100.0, 400.0, 250.0, 0.1)
        ambient_temperature = st.slider("Ambient_Temperature", 15.0, 30.0, 23.0, 0.1)
        machine_age = st.slider("Machine_Age", 0.0, 20.0, 7.0, 0.1)
        operator_experience = st.slider("Operator_Experience", 0.0, 100.0, 30.0, 0.1)
        maintenance_hours = st.slider("Maintenance_Hours", 0, 100, 50, 1)
        temperature_pressure_ratio = st.slider("Temperature_Pressure_Ratio", 1.0, 3.0, 1.8, 0.001)
        total_cycle_time = st.slider("Total_Cycle_Time", 25.0, 65.0, 50.0, 0.1)
        efficiency_score = st.slider("Efficiency_Score", 0.0, 1.0, 0.2, 0.001)
        machine_utilization = st.slider("Machine_Utilization", 0.0, 1.0, 0.5, 0.001)

        # Categorical inputs
        shift = st.selectbox("Shift", ["Day", "Evening", "Night"])
        machine_type = st.selectbox("Machine_Type", ["Type_A", "Type_B", "Type_C"])
        material_grade = st.selectbox("Material_Grade", ["Economy", "Standard", "Premium"])
        day_of_week = st.selectbox("Day_of_Week", [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ])

        submitted = st.form_submit_button("Predict")

    # Build raw input row in original training schema (include Timestamp placeholder if needed)
    user_row = {
        "Injection_Temperature": injection_temperature,
        "Injection_Pressure": injection_pressure,
        "Cycle_Time": cycle_time,
        "Cooling_Time": cooling_time,
        "Material_Viscosity": material_viscosity,
        "Ambient_Temperature": ambient_temperature,
        "Machine_Age": machine_age,
        "Operator_Experience": operator_experience,
        "Maintenance_Hours": maintenance_hours,
        "Shift": shift,
        "Machine_Type": machine_type,
        "Material_Grade": material_grade,
        "Day_of_Week": day_of_week,
        "Temperature_Pressure_Ratio": temperature_pressure_ratio,
        "Total_Cycle_Time": total_cycle_time,
        "Efficiency_Score": efficiency_score,
        "Machine_Utilization": machine_utilization,
    }

    return user_row, submitted


def main():
    st.title("Manufacturing Output Predictor")
    st.caption("Predict Parts_Per_Hour from manufacturing process settings and conditions")

    sidebar_info()

    try:
        model, scaler, encoder, model_name = load_artifacts()
        st.success(f"Loaded model: {model_name}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    user_row, submitted = build_input_form()

    if submitted:
        # Prepare features and predict
        try:
            X = preprocess_input(user_row, scaler=scaler, encoder=encoder)
            pred = float(model.predict(X)[0])
            st.subheader("Prediction")
            st.metric(label="Predicted Parts_Per_Hour", value=f"{pred:.2f}")
        except Exception as e:
            st.error(f"Failed to run prediction: {e}")


if __name__ == "__main__":
    main()


