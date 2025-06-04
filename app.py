import streamlit as st
from joblib import load

st.set_page_config(page_title="糖尿病预测系统", page_icon="🩺", layout="centered")

# 页面头部美化
st.markdown("""
    <style>
    .big-font {
        font-size:32px !important;
        color: #2E8B57;
        text-align: center;
    }
    </style>
    <p class="big-font">糖尿病预测系统 🧬</p>
    <hr style='border-top: 3px solid #bbb;' />
""", unsafe_allow_html=True)

# 模型加载
model = load("tree_diabetes_model.joblib")

# 输入区美化
st.subheader("请输入以下健康信息 👇")
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("怀孕次数", min_value=0)
    bp = st.number_input("血压")
    insulin = st.number_input("胰岛素")

with col2:
    glu = st.number_input("葡萄糖浓度")
    skin = st.number_input("皮肤厚度")
    bmi = st.number_input("BMI")
    
dpf = st.number_input("糖尿病家族史")
age = st.number_input("年龄", min_value=0)

if st.button("点击预测"):
    result = model.predict([[preg, glu, bp, skin, insulin, bmi, dpf, age]])[0]
    if result == 1:
        st.error("⚠️ 预测结果：可能有糖尿病")
    else:
        st.success("✅ 预测结果：无糖尿病")
