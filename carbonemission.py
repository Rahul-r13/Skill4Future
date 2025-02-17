import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Carbon Emission.csv")
    return df

df = load_data()

st.title("Carbon Emission Prediction Analysis")

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Handle missing values
df = df.assign(**{'Vehicle Type': df['Vehicle Type'].fillna('Unknown')})

# Convert categorical variables to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Ensure numeric columns are correctly formatted
df = df.apply(pd.to_numeric, errors='coerce')

# Check for missing values after processing
st.subheader("Missing Values After Preprocessing")
st.write(df.isnull().sum())

# Define features and target variable
X = df.drop(columns=['CarbonEmission'])  # Adjust column name if necessary
y = df['CarbonEmission']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.columns = X_train.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_test.columns = X_test.columns.str.replace(r"[\[\]<>]", "", regex=True)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

# Clean column names to remove special characters
X.columns = X.columns.str.replace(r"[\[\]<]", "", regex=True)

# Initialize models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}

for name, model in models.items():
    st.subheader(f"Training {name} Model")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test set

    # Compute evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results[name] = {"MAE": mae, "MSE": mse, "R2 Score": r2}

    st.write(f"{name} Performance:")
    st.write(f"MAE: {mae}")
    st.write(f"MSE: {mse}")
    st.write(f"R2 Score: {r2}")

# Plot Actual vs Predicted values for each model
st.subheader("Actual vs Predicted Values")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, model) in enumerate(models.items(), 1):
    y_pred = model.predict(X_test)
    
    axes[i-1].scatter(y_test, y_pred, alpha=0.5)
    axes[i-1].set_xlabel("Actual Carbon Emission")
    axes[i-1].set_ylabel("Predicted Carbon Emission")
    axes[i-1].set_title(f"{name} - Actual vs Predicted")

plt.tight_layout()
st.pyplot(fig)

# Convert results dictionary to DataFrame for visualization
results_df = pd.DataFrame(results).T

# Plot model performance comparison
st.subheader("Model Comparison - Performance Metrics")
fig2, ax = plt.subplots(figsize=(10, 5))
results_df.plot(kind='bar', ax=ax)
plt.title("Model Comparison - Performance Metrics")
plt.ylabel("Error / Score")
plt.xticks(rotation=0)
plt.legend(loc="upper right")
plt.grid()
st.pyplot(fig2)

st.success("Analysis Completed Successfully!")
