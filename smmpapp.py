import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import sys

# ====================== Fix: Compatible with older sklearn models ======================
# Fix No module named 'sklearn.ensemble._gb_losses'
class FixedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old loss function path
        if module == "sklearn.ensemble._gb_losses":
            module = "sklearn._losses"
        if module == "sklearn.ensemble.gradient_boosting":
            from sklearn.ensemble import gradient_boosting
            return getattr(gradient_boosting, name)
        return super().find_class(module, name)

# ====================== 1. Model Loading (Fixed Version) ======================
current_dir = Path(__file__).parent.resolve()
model_path = current_dir / "GBDT.pkl"

model = None
try:
    # Use compatible loader
    with open(model_path, 'rb') as f:
        unpickler = FixedUnpickler(f)
        model = unpickler.load()
    st.success(f"✅ Model loaded successfully! Path: {model_path}")
except Exception as e:
    st.error(f"❌ Model loading failed! Error: {str(e)}")
    st.warning("Please check if the GBDT.pkl file is complete or regenerate the model file")

# ====================== 2. Page UI ======================
st.write("# Severe Mycoplasma Pneumoniae Pneumonia(SMPP) Predictor")

# Feature order must be exactly the same as when training the model!
feature_names = ["COUGHday", "S100A8", "S100A8A9", "S100A9", "ALP", "LDH_L"]

# Input widgets
COUGHday = st.number_input("Enter the duration of cough (day)", 
                          min_value=1, max_value=120, value=10)
S100A8 = st.number_input("Enter your S100A8 protein value (pg/mL)", 
                        min_value=5, max_value=3500, value=300)
S100A9 = st.number_input("Enter your S100A9 protein value (pg/mL)", 
                        min_value=10, max_value=4000, value=300)
S100A8A9 = st.number_input("Enter your S100A8/A9 protein value (ug/mL)", 
                          min_value=0.05, max_value=20.0, value=2.5)
ALP = st.number_input("Enter your Alkaline Phosphatase (ALP) level (U/L)", 
                     min_value=10.5, max_value=1200.0, value=20.0)
LDH_L = st.number_input("Enter your Lactate Dehydrogenase (LDH) level (U/L)", 
                       min_value=10.0, max_value=3000.0, value=20.0)

# Construct input features
feature_values = [COUGHday, S100A8, S100A8A9, S100A9, ALP, LDH_L]
features = np.array([feature_values], dtype=np.float64)

# ====================== 3. Prediction Logic ======================
if st.button("Predict"):
    if model is None:
        st.error("❌ Prediction failed: Model not loaded, please check the GBDT.pkl file")
    elif not hasattr(model, "predict"):
        st.error("❌ Invalid model: Loaded object is not a valid sklearn model")
    else:
        try:
            # Core prediction
            Predicted_Degree = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]
            probability = predicted_proba[Predicted_Degree] * 100

            # Display results
            if Predicted_Degree == 1:
                st.success(f"✅ Prediction Result: Severe")
                st.write(f"**Severe Probability: {probability:.1f}%**")
            else:
                st.success(f"✅ Prediction Result: Mild")
                st.write(f"**Mild Probability: {probability:.1f}%**")

            st.write(f"Full Probability (Mild/Severe): {np.round(predicted_proba, 4)}")

            # ====================== 4. SHAP Visualization ======================
            try:
                explainer = shap.TreeExplainer(model)
                df_input = pd.DataFrame([feature_values], columns=feature_names)
                shap_values = explainer.shap_values(df_input)

                if isinstance(shap_values, list):
                    shap_val = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val = shap_values[0]
                    base_val = explainer.expected_value

                shap.force_plot(
                    base_val,
                    shap_val,
                    df_input,
                    matplotlib=True,
                    show=False
                )
                plt.tight_layout()
                plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
                plt.close()
                st.image("shap_force_plot.png", caption="SHAP Feature Contribution Explanation")
            except Exception as shap_err:
                st.warning(f"⚠️ SHAP plot generation failed: {str(shap_err)[:100]}")

        except Exception as pred_err:
            st.error(f"❌ Prediction failed: {str(pred_err)}")
            st.info(f"Debug info: Model type={type(model)}, Input shape={features.shape}")
