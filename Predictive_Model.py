# ============================================================
# House Price Prediction using Machine Learning
# Internship Project - Thiranex
#
# Submitted by: Anshul Patil
# Domain: Data Science
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.style.use("ggplot")

print("="*60)
print("Loading Dataset...")
print("="*60)

df = pd.read_csv("./dataset/Housing.csv")

print("Dataset Loaded Successfully!\n")

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInformation:")
df.info()

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDataset Shape after removing duplicates:")
print(df.shape)

encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = encoder.fit_transform(df[column])

print("\nCategorical columns converted successfully.")

# ============================================================
# DATA VISUALIZATION
# ============================================================

print("\nGenerating Visualizations...")

# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["price"], bins=30, kde=True)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.show()


# Average Price by Bedrooms
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="bedrooms", y="price", errorbar=None)
plt.title("Average House Price by Bedrooms")
plt.show()


# Average Price by Bathrooms
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="bathrooms", y="price", errorbar=None)
plt.title("Average House Price by Bathrooms")
plt.show()


# Area vs Price
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("Area vs House Price")
plt.show()


# Parking vs Price
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="parking", y="price")
plt.title("Parking vs House Price")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("Visualizations Completed.")

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

print("\n" + "="*60)
print("TRAINING LINEAR REGRESSION MODEL")
print("="*60)

# Features and Target
X = df.drop("price", axis=1)
y = df["price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions")

prediction_df = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred.astype(int)
})

print(prediction_df.head(10))

# ============================================================
# MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df.to_csv(
    "./dataset/Predicted_House_Prices.csv",
    index=False
)

print("\nPrediction file saved successfully!")

# ============================================================
# PROJECT SUMMARY
# ============================================================

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("""
Project Summary

✔ Dataset Loaded
✔ Data Preprocessed
✔ Visualizations Created
✔ Machine Learning Model Trained
✔ Predictions Generated
✔ Model Evaluated
✔ Prediction CSV Saved

Thank You!
""")

import joblib

joblib.dump(model, "./dataset/house_price_model.pkl")

print("Model saved successfully!")