import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
netflix = pd.read_csv(r"NFLX.csv")

if 'Adj Close' in netflix.columns:
    netflix.drop('Adj Close', axis=1, inplace=True)


netflix['Date'] = pd.to_datetime(netflix['Date'])
netflix['Year'] = netflix['Date'].dt.year
netflix['Month'] = netflix['Date'].dt.month
netflix['Day'] = netflix['Date'].dt.day
netflix.drop('Date', axis=1, inplace=True)


netflix['HL_diff'] = netflix['High'] - netflix['Low']
netflix['Price_range'] = netflix['High'] - netflix['Open']

# Define features and target
X = netflix.drop('Close', axis=1)
y = netflix['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models 
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),#regularization(prevents overfitting)
    "Lasso Regression": Lasso(alpha=0.01),#regularization and feature selection
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=2)
}

model_scores = {}
best_model = None
best_r2 = -np.inf

for name, model in models.items():
    if "Regression" in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    model_scores[name] = (model, r2)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# Streamlit UI
st.set_page_config(page_title="Netflix Stock Predictor", layout="centered")
st.title("Netflix Stock Close Price Predictor")

st.subheader("Model Performance Comparison")
for name, (_, r2) in model_scores.items():
    st.write(f"**{name}** → R² Score: `{r2:.4f}`")

st.success(f"Best Model: **{best_model_name}** (R² = {best_r2:.4f})")

st.subheader("Enter Stock Data")
Open = st.number_input("Open")
High = st.number_input("High")
Low = st.number_input("Low")
Volume = st.number_input("Volume")
Year = st.number_input("Year", min_value=2002, max_value=2023)
Month = st.number_input("Month", min_value=1, max_value=12)
Day = st.number_input("Day", min_value=1, max_value=31)

model_choice = st.selectbox("Select Model for Prediction", list(model_scores.keys()))


def predict(Open, High, Low, Volume, Year, Month, Day, model_choice):
    HL_diff = High - Low
    Price_range = High - Open

    input_dict = {
        'Open': Open,
        'High': High,
        'Low': Low,
        'Volume': Volume,
        'Year': Year,
        'Month': Month,
        'Day': Day,
        'HL_diff': HL_diff,
        'Price_range': Price_range
    }

    input_data = pd.DataFrame([input_dict])
    model, _ = model_scores[model_choice]

    if "Regression" in model_choice:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
    else:
        prediction = model.predict(input_data)

    return prediction[0]

if st.button("Predict Close Price"):
    result = predict(Open, High, Low, Volume, Year, Month, Day, model_choice)
    st.success(f"Predicted Close Price using **{model_choice}**: `{result:.2f}`")
