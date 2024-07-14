import streamlit as st
import pandas as pd
import pickle

st.title("Heart Disease Classification")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('heart.csv')

df = load_data()

# Data preprocessing
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
y = df['target'].values
X = df.drop('target', axis='columns')

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature names from training data
feature_names = X.columns

# Create input fields
st.header("Input Features")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=[0, 1])  # 0: Female, 1: Male
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
chol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3) Colored by Flourosopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# Prepare input for model prediction
input_data = pd.DataFrame({
    'age': [age], 'trestbps': [trestbps], 'chol': [chol], 'thalach': [thalach], 'oldpeak': [oldpeak],
    'sex_0': [1 if sex == 0 else 0], 'sex_1': [1 if sex == 1 else 0],
    'cp_0': [1 if cp == 0 else 0], 'cp_1': [1 if cp == 1 else 0], 'cp_2': [1 if cp == 2 else 0], 'cp_3': [1 if cp == 3 else 0],
    'fbs_0': [1 if fbs == 0 else 0], 'fbs_1': [1 if fbs == 1 else 0],
    'restecg_0': [1 if restecg == 0 else 0], 'restecg_1': [1 if restecg == 1 else 0], 'restecg_2': [1 if restecg == 2 else 0],
    'exang_0': [1 if exang == 0 else 0], 'exang_1': [1 if exang == 1 else 0],
    'slope_0': [1 if slope == 0 else 0], 'slope_1': [1 if slope == 1 else 0], 'slope_2': [1 if slope == 2 else 0],
    'ca_0': [1 if ca == 0 else 0], 'ca_1': [1 if ca == 1 else 0], 'ca_2': [1 if ca == 2 else 0], 'ca_3': [1 if ca == 3 else 0],
    'thal_0': [1 if thal == 0 else 0], 'thal_1': [1 if thal == 1 else 0], 'thal_2': [1 if thal == 2 else 0], 'thal_3': [1 if thal == 3 else 0]
})

# Ensure all feature columns are present
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Predict and display result
if st.button("Predict"):
    prediction = model.predict(input_data[feature_names])
    if prediction[0] == 1:
        st.write("The model predicts that the patient has heart disease.")
    else:
        st.write("The model predicts that the patient does not have heart disease.")
