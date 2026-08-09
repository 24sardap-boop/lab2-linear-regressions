# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Generate a Synthetic Housing/Price Dataset Offline
# This simulates house features (square footage, rooms, age, etc.) and continuous price targets
X_array, y_array = make_regression(n_samples=2000, n_features=5, noise=10.0, random_state=42)

# Convert to a pandas DataFrame for structured representation
df = pd.DataFrame(X_array, columns=['SqFootage', 'Rooms', 'Age', 'DistanceToCenter', 'IncomeLevel'])
df['Price'] = y_array + 250  # Shift prices to make them positive (representing house/car prices)

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# 2. Exploratory Data Analysis & Missing Value Verification
print("\nMissing values check:\n", df.isnull().sum())

# 3. Splitting Dataset into Features (X) and Target (y)
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining features shape: {X_train.shape}, Testing features shape: {X_test.shape}")

# 4. Feature Scaling for Linear Regression Optimization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Print Model Coefficients and Intercept
print("\nModel Intercept:", model.intercept_)
print("Model Coefficients:", model.coef_)

# 6. Evaluate Model Performance
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-Squared (R2 Score): {r2:.4f} ({r2 * 100:.2f}%)")