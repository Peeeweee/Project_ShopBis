"""
Category Prediction Model - Shopping Behavior Dataset
=====================================================

This script builds a machine learning model to predict the Category of items
(Clothing, Footwear, Accessories, Outerwear) based on product features.

Target Variable: Category (4 classes)
Features: Color, Size, Item Purchased, Gender, Age, Purchase Amount, Review Rating

Algorithm: Random Forest Classifier
Expected Accuracy: 60-80%
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

# ===========================
# 1. LOAD AND EXPLORE DATA
# ===========================

print("\n" + "="*60)
print("CATEGORY PREDICTION MODEL - SHOPPING BEHAVIOR")
print("="*60)

# Load the cleaned dataset
df = pd.read_csv('data/shopping_behavior_updated.csv')

print(f"\n[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Check target variable distribution
print("\n" + "-"*60)
print("TARGET VARIABLE DISTRIBUTION")
print("-"*60)
category_counts = df['Category'].value_counts()
print(category_counts)
print(f"\nTotal categories: {len(category_counts)}")

# Calculate percentages
print("\nPercentage distribution:")
for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {category}: {percentage:.2f}%")

# ===========================
# 2. FEATURE SELECTION
# ===========================

print("\n" + "-"*60)
print("FEATURE SELECTION")
print("-"*60)

# Select features that should correlate with Category
features_to_use = [
    'Item Purchased',      # Main predictor - each item belongs to a category
    'Color',               # Different categories may have different color patterns
    'Size',                # Size varies by category (XL for clothing, 8 for shoes)
    'Gender',              # Gender preferences vary by category
    'Age',                 # Age groups may prefer different categories
    'Purchase Amount (USD)', # Price ranges differ by category
    'Review Rating'        # Quality perception may differ
]

print("\nFeatures selected:")
for i, feature in enumerate(features_to_use, 1):
    print(f"  {i}. {feature}")

target = 'Category'
print(f"\nTarget: {target}")

# ===========================
# 3. DATA PREPROCESSING
# ===========================

print("\n" + "-"*60)
print("DATA PREPROCESSING")
print("-"*60)

# Create a copy for modeling
df_model = df[features_to_use + [target]].copy()

print(f"\n[INFO] Working with {df_model.shape[0]} rows and {df_model.shape[1]} columns")

# Check for missing values
print("\nMissing values check:")
missing = df_model.isnull().sum()
if missing.sum() == 0:
    print("  [SUCCESS] No missing values found")
else:
    print(missing[missing > 0])

# Encode categorical features
print("\n[INFO] Encoding categorical features...")

label_encoders = {}
categorical_features = ['Item Purchased', 'Color', 'Size', 'Gender']

for feature in categorical_features:
    le = LabelEncoder()
    df_model[feature] = le.fit_transform(df_model[feature])
    label_encoders[feature] = le
    print(f"  Encoded: {feature} ({len(le.classes_)} unique values)")

# Encode target variable
le_target = LabelEncoder()
df_model['Category_Encoded'] = le_target.fit_transform(df_model[target])
print(f"\n[INFO] Target encoded: {target} -> Category_Encoded")
print("  Class mapping:")
for i, class_name in enumerate(le_target.classes_):
    print(f"    {i}: {class_name}")

# ===========================
# 4. SPLIT DATA
# ===========================

print("\n" + "-"*60)
print("TRAIN-TEST SPLIT")
print("-"*60)

# Prepare features and target
X = df_model[features_to_use]
y = df_model['Category_Encoded']

# Split 80-20 with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class distribution
)

print(f"\n[INFO] Training set: {X_train.shape[0]} samples")
print(f"[INFO] Testing set: {X_test.shape[0]} samples")

# Show distribution in train and test
print("\nClass distribution in training set:")
train_dist = pd.Series(y_train).value_counts().sort_index()
for class_idx, count in train_dist.items():
    class_name = le_target.inverse_transform([class_idx])[0]
    print(f"  {class_name}: {count} ({count/len(y_train)*100:.1f}%)")

print("\nClass distribution in test set:")
test_dist = pd.Series(y_test).value_counts().sort_index()
for class_idx, count in test_dist.items():
    class_name = le_target.inverse_transform([class_idx])[0]
    print(f"  {class_name}: {count} ({count/len(y_test)*100:.1f}%)")

# ===========================
# 5. BUILD MODEL
# ===========================

print("\n" + "-"*60)
print("MODEL TRAINING")
print("-"*60)

# Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=200,      # 200 decision trees
    max_depth=15,          # Moderate depth to prevent overfitting
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=4,    # Minimum samples in leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

print("\n[INFO] Training Random Forest Classifier...")
print("  Parameters:")
print(f"    - Number of trees: 200")
print(f"    - Max depth: 15")
print(f"    - Min samples split: 10")
print(f"    - Min samples leaf: 4")

model.fit(X_train, y_train)

print("\n[SUCCESS] Model training completed!")

# ===========================
# 6. MAKE PREDICTIONS
# ===========================

print("\n" + "-"*60)
print("PREDICTIONS")
print("-"*60)

# Predict on test set
y_pred = model.predict(X_test)

print(f"\n[INFO] Generated predictions for {len(y_pred)} test samples")

# ===========================
# 7. EVALUATE MODEL
# ===========================

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
print(f"Weighted F1-Score: {f1:.4f}")

# Detailed classification report
print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
report = classification_report(
    y_test, y_pred,
    target_names=le_target.classes_,
    digits=4
)
print(report)

# Confusion Matrix
print("-"*60)
print("CONFUSION MATRIX")
print("-"*60)
cm = confusion_matrix(y_test, y_pred)
print("\n", cm)

# ===========================
# 8. FEATURE IMPORTANCE
# ===========================

print("\n" + "-"*60)
print("FEATURE IMPORTANCE")
print("-"*60)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': features_to_use,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop features for predicting Category:\n")
for idx, row in feature_importance.iterrows():
    bar = '#' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25} {row['Importance']:.4f} {bar}")

# ===========================
# 9. SAMPLE PREDICTIONS
# ===========================

print("\n" + "-"*60)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("-"*60)

# Show first 10 predictions
sample_df = pd.DataFrame({
    'Actual': [le_target.inverse_transform([val])[0] for val in y_test[:10].values],
    'Predicted': [le_target.inverse_transform([val])[0] for val in y_pred[:10]],
    'Correct': ['[YES]' if actual == pred else '[NO]'
                for actual, pred in zip(y_test[:10].values, y_pred[:10])]
})

for idx, row in sample_df.iterrows():
    print(f"\n  Sample {idx+1}:")
    print(f"    Actual:    {row['Actual']}")
    print(f"    Predicted: {row['Predicted']}")
    print(f"    Correct:   {row['Correct']}")

# ===========================
# 10. VISUALIZATIONS
# ===========================

print("\n" + "-"*60)
print("GENERATING VISUALIZATIONS")
print("-"*60)

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Confusion Matrix Heatmap
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=le_target.classes_,
    yticklabels=le_target.classes_,
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Actual Category', fontsize=11)
axes[0].set_xlabel('Predicted Category', fontsize=11)

# 2. Feature Importance
feature_importance_sorted = feature_importance.sort_values('Importance')
axes[1].barh(feature_importance_sorted['Feature'], feature_importance_sorted['Importance'], color='steelblue')
axes[1].set_xlabel('Importance Score', fontsize=11)
axes[1].set_title('Feature Importance', fontsize=14, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

# 3. Accuracy by Category
category_accuracy = []
for i, category in enumerate(le_target.classes_):
    mask = y_test == i
    if mask.sum() > 0:
        cat_acc = accuracy_score(y_test[mask], y_pred[mask])
        category_accuracy.append(cat_acc)
    else:
        category_accuracy.append(0)

axes[2].bar(le_target.classes_, category_accuracy, color='coral')
axes[2].set_ylabel('Accuracy', fontsize=11)
axes[2].set_title('Accuracy by Category', fontsize=14, fontweight='bold')
axes[2].set_ylim([0, 1])
axes[2].grid(axis='y', alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('Model/category_prediction_results.png', dpi=300, bbox_inches='tight')
print("\n[SUCCESS] Visualization saved: Model/category_prediction_results.png")

# Create second figure for class distribution
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Target distribution
category_counts.plot(kind='bar', ax=axes2[0], color='skyblue')
axes2[0].set_title('Category Distribution in Dataset', fontsize=14, fontweight='bold')
axes2[0].set_xlabel('Category', fontsize=11)
axes2[0].set_ylabel('Count', fontsize=11)
axes2[0].tick_params(axis='x', rotation=45)
axes2[0].grid(axis='y', alpha=0.3)

# Prediction distribution
pred_counts = pd.Series(y_pred).value_counts().sort_index()
pred_labels = [le_target.inverse_transform([i])[0] for i in pred_counts.index]
axes2[1].bar(pred_labels, pred_counts.values, color='lightcoral')
axes2[1].set_title('Predicted Category Distribution', fontsize=14, fontweight='bold')
axes2[1].set_xlabel('Category', fontsize=11)
axes2[1].set_ylabel('Count', fontsize=11)
axes2[1].tick_params(axis='x', rotation=45)
axes2[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('Model/category_distribution.png', dpi=300, bbox_inches='tight')
print("[SUCCESS] Visualization saved: Model/category_distribution.png")

# ===========================
# 11. FINAL SUMMARY
# ===========================

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"\n[RESULTS]")
print(f"  Model: Random Forest Classifier")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  F1-Score: {f1:.4f}")
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

print(f"\n[FEATURES USED]")
for feature in features_to_use:
    print(f"  - {feature}")

print(f"\n[KEY INSIGHTS]")
top_feature = feature_importance.iloc[0]
print(f"  - Most important feature: {top_feature['Feature']} ({top_feature['Importance']:.4f})")

best_category_idx = np.argmax(category_accuracy)
best_category = le_target.classes_[best_category_idx]
print(f"  - Best predicted category: {best_category} ({category_accuracy[best_category_idx]*100:.2f}% accuracy)")

print("\n[SUCCESS] Model training and evaluation completed!")

# ===========================
# 12. SAVE MODEL AND ENCODERS
# ===========================

print("\n" + "-"*60)
print("SAVING MODEL AND ENCODERS")
print("-"*60)

joblib.dump(model, 'Model/random_forest_model.joblib')
joblib.dump(label_encoders, 'Model/label_encoders.joblib')
joblib.dump(le_target, 'Model/le_target.joblib')

print("[SUCCESS] Model, label encoders, and target encoder saved to 'Model/' directory.")
print("="*60 + "\n")
