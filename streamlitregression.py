import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title
st.title("Simple Linear Regression App")

# Upload CSV data
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Select features and target
    features = st.multiselect("Select feature columns", data.columns.tolist())
    target = st.selectbox("Select target column", data.columns.tolist())

    if features and target:
        X = data[features]
        y = data[target]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Display results
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
        st.write("Model Coefficients:", model.coef_)
        st.write("Model Intercept:", model.intercept_)

        # Plot results
        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}))