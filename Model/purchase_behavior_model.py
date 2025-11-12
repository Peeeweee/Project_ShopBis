"""
Purchase Behavior Prediction Model - Shopping Behavior Dataset
===============================================================

This script builds a machine learning model to predict if a customer will buy again
based on their shopping behavior and demographics.

Target Variable: Will Buy Again (Yes/No)
Features: Age, Gender, Category, Season, Review Rating, Size, Shipping Type,
          Purchase Amount, Discount Applied, Promo Code Used

Algorithm: Random Forest Classifier
Expected Accuracy: 95%+
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import joblib
from pathlib import Path

# ===========================
# 1. LOAD AND EXPLORE DATA
# ===========================

print("\n" + "="*70)
print("PURCHASE BEHAVIOR PREDICTION MODEL - WILL CUSTOMER BUY AGAIN?")
print("="*70)

# Load the cleaned dataset
data_path = Path(__file__).parent.parent / 'data' / 'shopping_behavior_cleaned.csv'
df = pd.read_csv(data_path)

print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ===========================
# 2. CREATE TARGET VARIABLE
# ===========================

print("\n" + "-"*70)
print("CREATING TARGET VARIABLE: WILL BUY AGAIN")
print("-"*70)

# Create binary target based on "Previous Purchases"
# Logic: If previous purchases >= 2, customer will likely buy again
df['Will_Buy_Again'] = (df['Previous Purchases'] >= 2).astype(int)
df['Will_Buy_Again_Label'] = df['Will_Buy_Again'].map({0: 'No', 1: 'Yes'})

# Check distribution
target_counts = df['Will_Buy_Again_Label'].value_counts()
print("\nTarget Variable Distribution:")
print(target_counts)
print(f"\nPercentage distribution:")
for label, count in target_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {label}: {percentage:.2f}%")

# ===========================
# 3. FEATURE SELECTION
# ===========================

print("\n" + "-"*70)
print("FEATURE SELECTION")
print("-"*70)

# Select meaningful features for predicting repeat purchase behavior
features_to_use = [
    'Age',                      # Age demographics
    'Gender',                   # Gender preferences
    'Category',                 # Product category purchased
    'Season',                   # Shopping season
    'Review Rating',            # Customer satisfaction
    'Size',                     # Product size
    'Shipping Type',            # Shipping preference
    'Purchase Amount (USD)',    # Spending level
    'Discount Applied',         # Discount sensitivity
    'Promo Code Used',          # Promotion usage
]

print("\nFeatures selected:")
for i, feature in enumerate(features_to_use, 1):
    print(f"  {i}. {feature}")

target = 'Will_Buy_Again_Label'
print(f"\nTarget: {target}")

# ===========================
# 4. DATA PREPROCESSING
# ===========================

print("\n" + "-"*70)
print("DATA PREPROCESSING")
print("-"*70)

# Create a copy for modeling
df_model = df[features_to_use + [target]].copy()

print(f"\n[INFO] Working with {df_model.shape[0]} rows and {df_model.shape[1]} columns")

# Check for missing values
print("\nMissing values check:")
missing = df_model.isnull().sum()
if missing.sum() == 0:
    print("  [OK] No missing values found")
else:
    print(missing[missing > 0])

# Encode categorical variables
print("\n[INFO] Encoding categorical variables...")

label_encoders = {}
categorical_features = df_model.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove(target)  # Don't encode target yet

for feature in categorical_features:
    le = LabelEncoder()
    df_model[feature] = le.fit_transform(df_model[feature])
    label_encoders[feature] = le
    print(f"  [OK] Encoded: {feature}")

# Encode target variable separately
le_target = LabelEncoder()
y = le_target.fit_transform(df_model[target])
X = df_model.drop(target, axis=1)

print(f"\n[INFO] Feature matrix shape: {X.shape}")
print(f"[INFO] Target vector shape: {y.shape}")

# ===========================
# 5. TRAIN-TEST SPLIT
# ===========================

print("\n" + "-"*70)
print("SPLITTING DATA")
print("-"*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n[INFO] Training set: {X_train.shape[0]} samples")
print(f"[INFO] Testing set:  {X_test.shape[0]} samples")

# ===========================
# 6. MODEL TRAINING
# ===========================

print("\n" + "-"*70)
print("TRAINING RANDOM FOREST MODEL")
print("-"*70)

# Create and train the model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\n[INFO] Training model...")
model.fit(X_train, y_train)
print("  [OK] Model trained successfully!")

# ===========================
# 7. MODEL EVALUATION
# ===========================

print("\n" + "-"*70)
print("MODEL EVALUATION")
print("-"*70)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\n[RESULTS]")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  F1 Score: {f1:.4f}")

# Classification report
print("\n" + "-"*70)
print("CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Confusion Matrix
print("\n" + "-"*70)
print("CONFUSION MATRIX")
print("-"*70)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
print("\n" + "-"*70)
print("FEATURE IMPORTANCE")
print("-"*70)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']*100:.2f}%")

# ===========================
# 8. SAVE MODEL
# ===========================

print("\n" + "-"*70)
print("SAVING MODEL")
print("-"*70)

# Create saved_models directory if it doesn't exist
save_dir = Path(__file__).parent / 'saved_models'
save_dir.mkdir(exist_ok=True)

# Save model and encoders
model_path = save_dir / 'purchase_behavior_model.joblib'
encoders_path = save_dir / 'behavior_label_encoders.joblib'
target_encoder_path = save_dir / 'behavior_target_encoder.joblib'

joblib.dump(model, model_path)
joblib.dump(label_encoders, encoders_path)
joblib.dump(le_target, target_encoder_path)

print(f"\n[INFO] Model saved to: {model_path}")
print(f"[INFO] Encoders saved to: {encoders_path}")
print(f"[INFO] Target encoder saved to: {target_encoder_path}")

# ===========================
# 9. VISUALIZATIONS
# ===========================

print("\n" + "-"*70)
print("CREATING VISUALIZATIONS")
print("-"*70)

# Create visualizations directory
viz_dir = Path(__file__).parent / 'visualizations'
viz_dir.mkdir(exist_ok=True)

# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
plt.title('Purchase Behavior Prediction - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(viz_dir / 'behavior_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: behavior_confusion_matrix.png")
plt.close()

# 2. Feature Importance Chart
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
sns.barplot(data=top_features, y='Feature', x='Importance', palette='Oranges_r')
plt.title('Top 10 Features for Predicting Purchase Behavior', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(viz_dir / 'behavior_feature_importance.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: behavior_feature_importance.png")
plt.close()

# 3. Target Distribution
plt.figure(figsize=(8, 6))
target_counts.plot(kind='bar', color=['#DD0303', '#FA812F'])
plt.title('Customer Purchase Behavior Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Will Buy Again?', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(viz_dir / 'behavior_distribution.png', dpi=300, bbox_inches='tight')
print("  [OK] Saved: behavior_distribution.png")
plt.close()

# ===========================
# 10. SUMMARY
# ===========================

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n[OK] Model Accuracy: {accuracy*100:.2f}%")
print(f"[OK] Features Used: {len(features_to_use)}")
print(f"[OK] Training Samples: {len(X_train)}")
print(f"[OK] Test Samples: {len(X_test)}")
print(f"\n[OK] Model saved successfully!")
print(f"[OK] Ready for predictions in dashboard!")
print("\n" + "="*70)
