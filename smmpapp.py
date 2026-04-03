import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# ====================== 核心修复：确保模型正确加载并验证 ======================
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 兼容所有GBDT相关类
        gbdt_classes = {
            'GradientBoostingClassifier': GradientBoostingClassifier,
            'GradientBoostingRegressor': GradientBoostingRegressor
        }
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # 如果找不到，返回原生GBDT类
            return gbdt_classes.get(name, GradientBoostingClassifier)

# 初始化模型变量
model = None
current_dir = Path(__file__).parent.resolve()

# 尝试加载模型并验证
try:
    # 尝试1：models文件夹下的模型
    model_path = current_dir / "models" / "GBDT.pkl"
    with open(model_path, 'rb') as f:
        model = SafeUnpickler(f).load()
    st.success(f"✅ Model loaded from: {model_path}")
except Exception as e1:
    try:
        # 尝试2：根目录下的模型
        model_path = current_dir / "GBDT.pkl"
        with open(model_path, 'rb') as f:
            model = SafeUnpickler(f).load()
        st.success(f"✅ Model loaded from: {model_path}")
    except Exception as e2:
        st.error(f"❌ Failed to load model: {str(e2)}")
        st.warning("Please check if GBDT.pkl exists in root directory or models/ folder!")

# 验证模型是否有效
if model is not None:
    # 检查模型是否有predict方法
    if not hasattr(model, 'predict'):
        st.error("❌ Loaded model is invalid (no predict method)!")
        # 备用方案：创建一个空的GBDT模型（避免崩溃）
        model = GradientBoostingClassifier()
        st.warning("⚠️ Using fallback GBDT model (not trained)!")
else:
    # 模型加载失败时，创建空模型避免崩溃
    model = GradientBoostingClassifier()
    st.warning("⚠️ Using fallback GBDT model (not trained)!")

# ====================== 页面UI ======================
st.write("# Severe Mycoplasma Pneumoniae Pneumonia(SMPP) Predictor")

feature_names = ["COUGHday", "S100A8", "S100A8A9", "S100A9", "ALP", "LDH_L"]

# 输入控件
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

feature_values = [COUGHday, S100A8, S100A8A9, S100A9, ALP, LDH_L]
# 确保输入格式正确（二维数组）
features = np.array([feature_values], dtype=np.float64)

# ====================== 预测逻辑（增加完整异常捕获） ======================
if st.button("Predict"):
    try:
        # 核心预测（增加异常捕获）
        Predicted_Degree = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[Predicted_Degree] * 100

        # 显示结果
        if Predicted_Degree == 1:
            st.success(f"✅ Prediction Result: Severe")
            st.write(f"**Probability of Severe SMMP:** {probability:.1f}%")
        else:
            st.success(f"✅ Prediction Result: Mild")
            st.write(f"**Probability of Mild SMMP:** {probability:.1f}%")

        st.write(f"Full Probabilities (Mild/Severe): {predicted_proba.round(4)}")

        # SHAP可视化（增加更强的异常处理）
        try:
            explainer = shap.TreeExplainer(model)
            df = pd.DataFrame([feature_values], columns=feature_names)
            shap_values = explainer.shap_values(df)

            # 兼容不同版本的SHAP输出格式
            if isinstance(shap_values, list):
                shap_val = shap_values[1][0]  # 取重症类的SHAP值
                base_val = explainer.expected_value[1]
            else:
                shap_val = shap_values[0]
                base_val = explainer.expected_value

            # 生成SHAP图
            shap.force_plot(
                base_val,
                shap_val,
                df,
                matplotlib=True,
                show=False  # 避免自动弹出窗口
            )
            plt.tight_layout()
            plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
            plt.close()
            st.image("shap_force_plot.png", caption="SHAP Explanation Plot")
        except Exception as shap_err:
            st.warning(f"⚠️ SHAP plot generation failed: {str(shap_err)[:100]}")

    except Exception as pred_err:
        st.error(f"❌ Prediction failed: {str(pred_err)}")
        # 显示详细错误信息（帮助排查）
        st.info(f"Debug info: model type = {type(model)}, features shape = {features.shape}")
