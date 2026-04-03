import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

# ====================== 修复1：解决模型加载 ModuleNotFound 错误 ======================
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 强制从 sklearn 加载 GBDT 类，解决云端找不到模块的问题
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier

# ====================== 修复2：正确路径加载模型 ======================
current_dir = Path(__file__).parent.resolve()

# 两种路径兼容：models/GBDT.pkl 或 根目录 GBDT.pkl
try:
    model_path = current_dir / "models" / "GBDT.pkl"
    with open(model_path, 'rb') as f:
        model = SafeUnpickler(f).load()
except:
    model_path = current_dir / "GBDT.pkl"
    with open(model_path, 'rb') as f:
        model = SafeUnpickler(f).load()

# ====================== 页面 ======================
st.write("# Severe Mycoplasma Pneumoniae Pneumonia(SMPP) Predictor")

feature_names = ["COUGHday", "S100A8", "S100A8A9", "S100A9", "ALP", "LDH_L"]

COUGHday = st.number_input("Enter the duration of cough (day)", min_value=1, max_value=120, value=10)
S100A8 = st.number_input("Enter your S100A8 protein value (pg/mL)", min_value=5, max_value=3500, value=300)
S100A9 = st.number_input("Enter your S100A9 protein value (pg/mL)", min_value=10, max_value=4000, value=300)
S100A8A9 = st.number_input("Enter your S100A8/A9 protein value (ug/mL)", min_value=0.05, max_value=20.0, value=2.5)
ALP = st.number_input("Enter your Alkaline Phosphatase (ALP) level (U/L)", min_value=10.5, max_value=1200.0, value=20.0)
LDH_L = st.number_input("Enter your Lactate Dehydrogenase (LDH) level (U/L)", min_value=10.0, max_value=3000.0, value=20.0)

feature_values = [COUGHday, S100A8, S100A8A9, S100A9, ALP, LDH_L]
features = np.array([feature_values])

# ====================== 预测 ======================
if st.button("Predict"):
    Predicted_Degree = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[Predicted_Degree] * 100

    if Predicted_Degree == 1:
        st.success(f"✅ Prediction Result: Severe")
        st.write(f"**Probability of Severe SMMP:** {probability:.1f}%")
    else:
        st.success(f"✅ Prediction Result: Mild")
        st.write(f"**Probability of Mild SMMP:** {probability:.1f}%")

    st.write(f"Full Probabilities: {predicted_proba}")

    # ====================== SHAP 解释图 ======================
    try:
        explainer = shap.TreeExplainer(model)
        df = pd.DataFrame([feature_values], columns=feature_names)
        shap_values = explainer.shap_values(df)

        shap.force_plot(
            explainer.expected_value[1],  # 取1类（重症）的基线
            shap_values[1][0],
            df,
            matplotlib=True
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
        plt.close()
        st.image("shap_force_plot.png", caption="SHAP Explanation Plot")
    except Exception as e:
        st.warning(f"SHAP figure generation failed: {str(e)}")
