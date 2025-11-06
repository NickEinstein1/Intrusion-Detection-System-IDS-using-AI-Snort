"""
Multi-class Classification Script for Intrusion Detection System
Trains a Random Forest classifier to identify specific attack types.

Attack Categories (10 classes):
1. Normal
2. Fuzzers
3. Analysis
4. Backdoors
5. DoS
6. Exploits
7. Generic
8. Reconnaissance
9. Shellcode
10. Worms
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("MULTI-CLASS INTRUSION DETECTION SYSTEM")
print("="*70)
print()

# ============================================================================
# 1. LOAD ORIGINAL DATA
# ============================================================================
print("Step 1: Loading UNSW-NB15 dataset...")
print("-" * 70)

# Try to load the data from common locations
data_paths = [
    r"C:\Users\User\Documents\Cybersecurity\IDS\IDS\Data\UNSW_NB15_training-set.csv",
    "UNSW_NB15_training-set.csv",
    "Data/UNSW_NB15_training-set.csv",
    "../Data/UNSW_NB15_training-set.csv"
]

train_data = None
test_data = None

for path in data_paths:
    try:
        train_data = pd.read_csv(path)
        test_path = path.replace("training-set", "testing-set")
        test_data = pd.read_csv(test_path)
        print(f"✓ Data loaded from: {path}")
        break
    except FileNotFoundError:
        continue

if train_data is None:
    print("✗ Error: Could not find UNSW-NB15 dataset files.")
    print("\nPlease ensure the dataset files are in one of these locations:")
    for path in data_paths:
        print(f"  - {path}")
    print("\nOr update the path in this script.")
    exit(1)

print(f"✓ Training data shape: {train_data.shape}")
print(f"✓ Test data shape: {test_data.shape}")
print()

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("Step 2: Preprocessing data for multi-class classification...")
print("-" * 70)

# Keep attack_cat column (don't drop it this time!)
# Drop only the id column
train_data = train_data.drop(columns=["id"])
test_data = test_data.drop(columns=["id"])

# Also drop the binary label column since we're using attack_cat
if 'label' in train_data.columns:
    train_data = train_data.drop(columns=["label"])
    test_data = test_data.drop(columns=["label"])

print(f"✓ Columns after dropping id and label: {train_data.shape[1]}")

# Check attack categories
attack_categories = train_data['attack_cat'].value_counts()
print(f"\nAttack categories found: {len(attack_categories)}")
for category, count in attack_categories.items():
    print(f"  {category:20s}: {count:6d}")
print()

# Identify categorical columns (excluding attack_cat which is our target)
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('attack_cat')  # Remove target from features

print(f"Categorical feature columns: {len(categorical_cols)}")
print(f"  {categorical_cols}")
print()

# Encode categorical features
print("Encoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    train_data[col] = label_encoders[col].fit_transform(train_data[col])
    
    # Map test data using the same encoder, replacing unknown labels with -1
    test_data[col] = test_data[col].apply(
        lambda x: label_encoders[col].classes_.tolist().index(x) 
        if x in label_encoders[col].classes_ else -1
    )

print("✓ Categorical features encoded")

# Encode target variable (attack_cat)
print("\nEncoding target variable (attack_cat)...")
attack_encoder = LabelEncoder()
train_data['attack_cat_encoded'] = attack_encoder.fit_transform(train_data['attack_cat'])
test_data['attack_cat_encoded'] = test_data['attack_cat'].apply(
    lambda x: attack_encoder.classes_.tolist().index(x) 
    if x in attack_encoder.classes_ else -1
)

# Save the encoder for later use
joblib.dump(attack_encoder, 'attack_category_encoder.pkl')
print(f"✓ Target variable encoded into {len(attack_encoder.classes_)} classes")
print(f"✓ Class mapping saved to 'attack_category_encoder.pkl'")
print()

# Display class mapping
print("Class Mapping:")
for idx, category in enumerate(attack_encoder.classes_):
    print(f"  {idx}: {category}")
print()

# Normalize numerical features
print("Normalizing numerical features...")
numerical_cols = train_data.columns.difference(categorical_cols).difference(['attack_cat', 'attack_cat_encoded'])
scaler = StandardScaler()
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols])
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols])
print(f"✓ {len(numerical_cols)} numerical features normalized")
print()

# Save preprocessed data
train_data.to_csv('train_multiclass_processed.csv', index=False)
test_data.to_csv('test_multiclass_processed.csv', index=False)
print("✓ Preprocessed data saved")
print()

# ============================================================================
# 3. PREPARE FEATURES AND LABELS
# ============================================================================
print("Step 3: Preparing features and labels...")
print("-" * 70)

X_train = train_data.drop(['attack_cat', 'attack_cat_encoded'], axis=1)
y_train = train_data['attack_cat_encoded']
X_test = test_data.drop(['attack_cat', 'attack_cat_encoded'], axis=1)
y_test = test_data['attack_cat_encoded']

print(f"✓ Training features: {X_train.shape}")
print(f"✓ Training labels: {y_train.shape}")
print(f"✓ Test features: {X_test.shape}")
print(f"✓ Test labels: {y_test.shape}")
print()

# Check class distribution
print("Class distribution in training set:")
class_dist = y_train.value_counts().sort_index()
for class_idx, count in class_dist.items():
    class_name = attack_encoder.classes_[class_idx]
    percentage = (count / len(y_train)) * 100
    print(f"  {class_idx} ({class_name:20s}): {count:6d} ({percentage:5.2f}%)")
print()

# ============================================================================
# 4. TRAIN MULTI-CLASS MODEL
# ============================================================================
print("Step 4: Training multi-class Random Forest classifier...")
print("-" * 70)

# Use class_weight='balanced' to handle class imbalance
print("Using class_weight='balanced' to handle class imbalance...")
print("Training model (this may take several minutes)...")
print()

start_time = time.time()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train, y_train)

elapsed_time = time.time() - start_time
print(f"\n✓ Model trained in {elapsed_time/60:.2f} minutes")
print()

# Save the model
joblib.dump(model, 'intrusion_detection_model_multiclass.pkl')
print("✓ Model saved as 'intrusion_detection_model_multiclass.pkl'")
print()

# ============================================================================
# 5. EVALUATE MODEL
# ============================================================================
print("Step 5: Evaluating multi-class model...")
print("-" * 70)

# Make predictions
y_pred = model.predict(X_test)

# Overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

# Per-class metrics
print("Per-Class Metrics:")
print("=" * 70)
print(f"{'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
print("-" * 70)

# Calculate per-class metrics
for class_idx in range(len(attack_encoder.classes_)):
    class_name = attack_encoder.classes_[class_idx]
    
    # Binary classification for this class vs all others
    y_test_binary = (y_test == class_idx).astype(int)
    y_pred_binary = (y_pred == class_idx).astype(int)
    
    if y_test_binary.sum() > 0:  # Only if class exists in test set
        precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
        support = y_test_binary.sum()
        
        print(f"{class_name:<20s} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10d}")

print()

# Detailed classification report
print("Detailed Classification Report:")
print("=" * 70)
print(classification_report(y_test, y_pred, target_names=attack_encoder.classes_, zero_division=0))

# Save classification report
report_dict = classification_report(y_test, y_pred, target_names=attack_encoder.classes_, 
                                   output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv('multiclass_classification_report.csv')
print("✓ Classification report saved to 'multiclass_classification_report.csv'")
print()

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("Step 6: Saving results...")
print("-" * 70)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=attack_encoder.classes_, columns=attack_encoder.classes_)
cm_df.to_csv('multiclass_confusion_matrix.csv')
print("✓ Confusion matrix saved to 'multiclass_confusion_matrix.csv'")

# Save feature names for later use
feature_names = X_train.columns.tolist()
pd.DataFrame({'feature': feature_names}).to_csv('multiclass_feature_names.csv', index=False)
print("✓ Feature names saved to 'multiclass_feature_names.csv'")
print()

print("="*70)
print("MULTI-CLASS CLASSIFICATION COMPLETE!")
print("="*70)
print()
print("Next: Run 'python multiclass_visualizations.py' to generate visualizations")

