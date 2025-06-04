import pandas as pd
from joblib import load

# 加载保存的模型
knn = load('knn_diabetes_model.joblib')
tree = load('tree_diabetes_model.joblib')

# 预测输入的数据（怀孕次数、血糖水平、血压、皮肤厚度、胰岛素、BMI、亲属病史、年龄）
test_data = pd.DataFrame({
    'Pregnancies': [3],
    'Glucose': [126],
    'BloodPressure': [88],
    'SkinThickness': [41],
    'Insulin': [235],
    'BMI': [39.3],
    'DiabetesPedigreeFunction': [0.704],
    'Age': [27]
})

expected_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

# 进行预测
prediction = tree.predict(test_data)
probability = tree.predict_proba(test_data)

# 输出结果
print("预测结果:", prediction)
print("患病概率:", probability[0][1])
