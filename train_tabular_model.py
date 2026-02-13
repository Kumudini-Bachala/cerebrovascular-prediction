import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

# --- Construct absolute path to the CSV file ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'stroke.csv')

# --- 1. Load Data ---
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Error: 'stroke.csv' not found at {csv_path}")
    exit()

# --- 2. Initial Data Cleanup ---
# Drop the 'id' column as it's not a predictive feature
df = df.drop('id', axis=1)

# The 'Other' gender category has only one instance and can be dropped for simplicity
df = df[df['gender'] != 'Other']

# For BMI, we will impute missing values instead of dropping rows
# This will be handled by the SimpleImputer in our pipeline

# --- 3. Define Features (X) and Target (y) ---
X = df.drop('stroke', axis=1)
y = df['stroke']

# --- 4. Identify Column Types for Preprocessing ---
# Identifying categorical and numerical columns
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_features = ['age', 'avg_glucose_level', 'bmi']

# Ensure all feature names are strings
categorical_features = [str(col) for col in categorical_features]
numerical_features = [str(col) for col in numerical_features]

# --- 5. Create Preprocessing Pipelines ---
# Pipeline for numerical features: impute missing values (e.g., for BMI) with the median, then scale.
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: impute missing values with the most frequent value, then one-hot encode.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# --- 6. Create a ColumnTransformer to Apply Different Transformations ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep other columns (if any), though we've defined all
)

# --- 7. Create the Full Model Pipeline ---
# This pipeline first preprocesses the data, then trains the logistic regression model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', random_state=42))
])

# --- 8. Split Data and Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the stroke risk prediction model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 9. Evaluate the Model ---
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# --- 10. Save the Trained Model Pipeline ---
# The saved file contains the entire pipeline, including preprocessing and the model itself
# Save the file in the same directory as the script
save_path = os.path.join(script_dir, 'stroke_risk_model.joblib')
joblib.dump(model_pipeline, save_path)
print(f"Model pipeline saved to: {save_path}")

