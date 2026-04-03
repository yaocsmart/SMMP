import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# ====================== 1. 模型加载（完全适配你的GitHub结构） ======================
# 直接定位根目录的GBDT.pkl，不做多余路径判断
current_dir = Path(__file__).parent.resolve()
model_path = current_dir / "GBDT.pkl"

# 强制加载模型，增加详细错误提示
model = None
try:
    # 直接用joblib加载（你的模型是sklearn原生，无需自定义Unpickler）
    # model = joblib.load(model_path)
    with open("GBDT.pkl", 'rb') as f:
         model = pickle.load(f)
    st.success(f"✅ 模型加载成功！路径：{model_path}")
except Exception as e:
    st.error(f"❌ 模型加载失败！错误信息：{str(e)}")
    st.warning("请检查GBDT.pkl文件是否完整，或重新生成模型文件")

# ====================== 2. 页面UI ======================
st.write("# Severe Mycoplasma Pneumoniae Pneumonia(SMPP) Predictor")

# 特征顺序必须和训练模型时完全一致！
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

# 构造输入特征（严格二维数组，sklearn要求）
feature_values = [COUGHday, S100A8, S100A8A9, S100A9, ALP, LDH_L]
features = np.array([feature_values], dtype=np.float64)

# ====================== 3. 预测逻辑（增加模型有效性校验） ======================
if st.button("Predict"):
    # 先校验模型是否有效
    if model is None:
        st.error("❌ 无法预测：模型加载失败，请检查GBDT.pkl文件")
    elif not hasattr(model, "predict"):
        st.error("❌ 模型无效：加载的对象不是可预测的sklearn模型")
    else:
        try:
            # 核心预测
            Predicted_Degree = model.predict(features)[0]
            predicted_proba = model.predict_proba(features)[0]
            probability = predicted_proba[Predicted_Degree] * 100

            # 显示结果
            if Predicted_Degree == 1:
                st.success(f"✅ 预测结果：重症 (Severe)")
                st.write(f"**重症概率：{probability:.1f}%**")
            else:
                st.success(f"✅ 预测结果：轻症 (Mild)")
                st.write(f"**轻症概率：{probability:.1f}%**")

            st.write(f"完整概率（轻症/重症）：{np.round(predicted_proba, 4)}")

            # ====================== 4. SHAP可视化（兼容所有版本） ======================
            try:
                explainer = shap.TreeExplainer(model)
                df_input = pd.DataFrame([feature_values], columns=feature_names)
                shap_values = explainer.shap_values(df_input)

                # 兼容不同版本SHAP输出
                if isinstance(shap_values, list):
                    # 二分类模型：shap_values[0]是负类，shap_values[1]是正类（重症=1）
                    shap_val = shap_values[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_val = shap_values[0]
                    base_val = explainer.expected_value

                # 生成SHAP力图
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
                st.image("shap_force_plot.png", caption="SHAP特征贡献解释图")
            except Exception as shap_err:
                st.warning(f"⚠️ SHAP图生成失败：{str(shap_err)[:100]}")

        except Exception as pred_err:
            st.error(f"❌ 预测失败：{str(pred_err)}")
            st.info(f"调试信息：模型类型={type(model)}, 输入形状={features.shape}")
