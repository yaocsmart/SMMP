import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
model = joblib.load('GBDT.pkl')
st.write("# Severe Mycoplasma Pneumoniae Pneumonia(SMPP) Predictor")
feature_names = [ "COUGHday", "S100A8", "S100A8A9", "S100A9", "ALP",  "LDH_L"]
COUGHday = st.number_input("Enter the duration of cough (day) ",min_value=1, max_value=120, value=10)
S100A8 = st.number_input("Enter your S100A8 protein value (pg/mL) ",min_value=5, max_value=3500, value=300)
S100A9 = st.number_input("Enter your S100A9 protein value (pg/mL) ", min_value=10, max_value=4000, value=300)
S100A8A9 = st.number_input("Enter your S100A8/A9 protein value (ug/mL) ", min_value=0.05, max_value=20.0, value=2.5)
ALP = st.number_input("Enter your Alkaline Phosphatase(ALP) level (U/L) ", min_value=10.5, max_value=1200.0, value=20.0)
LDH_L = st.number_input("Enter your Lactate Dehydrogenase(LDH) level (U/L) ", min_value=10.0, max_value=3000.0, value=20.0)
feature_values = [COUGHday, S100A8, S100A8A9, S100A9, ALP,  LDH_L]
features = np.array([feature_values])
if st.button("Predict"):
   Predicted_Degree = model.predict(features)[0]
   predicted_proba = model.predict_proba(features)[0]
    
   probability = predicted_proba[Predicted_Degree] * 100
   if Predicted_Degree == 1:
         st.write(f"Predicted Degree Severe")
         st.write(f"**Prediction Probabilities:** {predicted_proba}")
         st.write( f"According to our model, you have a high risk of SMMP. ")
         st.write(f"The model predicts that your probability of having SMMP disease is {probability:.1f}%. ")
   else:
        st.write(f"Predicted Degree Mild")
        st.write(f"**Prediction Probabilities:** {predicted_proba}")
        st.write( f"According to our model, you have a high risk of SMMP. ")
        st.write( f"The model predicts that your probability of not having SMMP disease is {probability:.1f}%. ")

   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

   shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
   plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

   st.image("shap_force_plot.png")


