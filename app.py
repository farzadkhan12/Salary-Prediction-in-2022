import joblib
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor


# --- Preprocessing setup ---
numeric_pipe = Pipeline([
    ("Scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
])

transform = ColumnTransformer([
    ("numeric", numeric_pipe, ["YearsCodePro"]),
    ("cat", cat_pipe, ["EdLevel", "Country"])
])

# --- Load trained model ---
model = joblib.load("gbr_n.joblib")

# --- Streamlit UI ---
st.title("💼 Salary Prediction (2022)")

columns = ['Country', 'EdLevel', 'YearsCodePro']

countries = [
    "United States of America",
    "Iran, Islamic Republic of...",
    "India",
    "United Kingdom of Great Britain and Northern Ireland",
    "Germany",
    "Canada",
    "Brazil",
    "France",
    "Spain",
    "Australia",
    "Netherlands",
    "Poland",
    "Italy",
    "Russian Federation",
    "Sweden",
    "Switzerland",
    "Palestine",
    "Austria",
    "Portugal",
    "Denmark",
    "Turkey",
    "Belgium",
    "Norway",
    "Finland",
    "Greece",
    "Czech Republic",
    "New Zealand",
    "Mexico",
    "South Africa",
    "Pakistan"
]

education_levels = (
    "Less than a Bachelors",
    "Bachelor’s degree",
    "Master’s degree"
)

country = st.selectbox("🌍 Country", countries)
education = st.selectbox("🎓 Education Level", education_levels)
experience = st.slider("🧠 Years of Experience", 0, 50, 1)

# --- Predict button ---
if st.button("🚀 Predict!"):
    # ساخت ورودی جدید
    x_new = np.array([country, education, experience])
    X_new_df = pd.DataFrame([x_new], columns=columns)

    # اجرای transform روی داده ورودی
    X_new_prepared = transform.fit_transform(X_new_df)

    # پیش‌بینی
    salary = model.predict(X_new_prepared)

    st.subheader(f"💰 Estimated Salary: **${salary[0]:,.2f}**")
