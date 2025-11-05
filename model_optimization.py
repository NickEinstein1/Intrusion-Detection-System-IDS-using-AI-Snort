"""
Model Optimization Script for Intrusion Detection System
Implements cross-validation and hyperparameter tuning for Random Forest classifier.

This script:
1. Performs K-Fold Cross-Validation to evaluate model stability
2. Uses GridSearchCV to find optimal hyperparameters
3. Compares baseline vs optimized model performance
4. Generates performance comparison visualizations
5. Saves the optimized model
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("INTRUSION DETECTION SYSTEM - MODEL OPTIMIZATION")
print("="*70)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("Step 1: Loading preprocessed data...")
try:
    train_data = pd.read_csv('train_processed.csv')
    test_data = pd.read_csv('test_processed.csv')
    print(f"✓ Training data loaded: {train_data.shape}")
    print(f"✓ Test data loaded: {test_data.shape}")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run the main notebook first to generate preprocessed data.")
    exit(1)

# Separate features and target
X_train = train_data.drop('attack_cat', axis=1)
y_train = train_data['attack_cat']
X_test = test_data.drop('attack_cat', axis=1)
y_test = test_data['attack_cat']

print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print()

# ============================================================================
# 2. BASELINE MODEL EVALUATION
# ============================================================================
print("Step 2: Evaluating baseline model...")
print("-" * 70)

# Load existing model or train a baseline
try:
    baseline_model = joblib.load('intrusion_detection_model_unsw.pkl')
    print("✓ Loaded existing baseline model")
except FileNotFoundError:
    print("Training new baseline model...")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    baseline_model.fit(X_train, y_train)
    print("✓ Baseline model trained")

# Baseline predictions
baseline_pred = baseline_model.predict(X_test)
baseline_pred_proba = baseline_model.predict_proba(X_test)[:, 1]

# Baseline metrics
baseline_metrics = {
    'accuracy': accuracy_score(y_test, baseline_pred),
    'precision': precision_score(y_test, baseline_pred),
    'recall': recall_score(y_test, baseline_pred),
    'f1': f1_score(y_test, baseline_pred),
    'roc_auc': roc_auc_score(y_test, baseline_pred_proba)
}

print("\nBaseline Model Performance:")
print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
print(f"  Precision: {baseline_metrics['precision']:.4f}")
print(f"  Recall:    {baseline_metrics['recall']:.4f}")
print(f"  F1-Score:  {baseline_metrics['f1']:.4f}")
print(f"  ROC-AUC:   {baseline_metrics['roc_auc']:.4f}")
print()

# ============================================================================
# 3. K-FOLD CROSS-VALIDATION
# ============================================================================
print("Step 3: Performing K-Fold Cross-Validation...")
print("-" * 70)

# Use StratifiedKFold to maintain class distribution
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Running 5-Fold Cross-Validation on baseline model...")
cv_scores = cross_val_score(baseline_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"\nCross-Validation Results:")
print(f"  Fold Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"  Min CV Accuracy:  {cv_scores.min():.4f}")
print(f"  Max CV Accuracy:  {cv_scores.max():.4f}")
print()

# ============================================================================
# 4. HYPERPARAMETER TUNING WITH GRID SEARCH
# ============================================================================
print("Step 4: Hyperparameter Tuning with GridSearchCV...")
print("-" * 70)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

print(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
print("This may take several minutes...")
print()

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,  # Use 3-fold for faster computation
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

# Perform grid search
start_time = time.time()
grid_search.fit(X_train, y_train)
elapsed_time = time.time() - start_time

print(f"\n✓ Grid Search completed in {elapsed_time/60:.2f} minutes")
print(f"\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\nBest Cross-Validation Score: {grid_search.best_score_:.4f}")
print()

# ============================================================================
# 5. OPTIMIZED MODEL EVALUATION
# ============================================================================
print("Step 5: Evaluating optimized model...")
print("-" * 70)

# Get the best model
optimized_model = grid_search.best_estimator_

# Optimized predictions
optimized_pred = optimized_model.predict(X_test)
optimized_pred_proba = optimized_model.predict_proba(X_test)[:, 1]

# Optimized metrics
optimized_metrics = {
    'accuracy': accuracy_score(y_test, optimized_pred),
    'precision': precision_score(y_test, optimized_pred),
    'recall': recall_score(y_test, optimized_pred),
    'f1': f1_score(y_test, optimized_pred),
    'roc_auc': roc_auc_score(y_test, optimized_pred_proba)
}

print("\nOptimized Model Performance:")
print(f"  Accuracy:  {optimized_metrics['accuracy']:.4f}")
print(f"  Precision: {optimized_metrics['precision']:.4f}")
print(f"  Recall:    {optimized_metrics['recall']:.4f}")
print(f"  F1-Score:  {optimized_metrics['f1']:.4f}")
print(f"  ROC-AUC:   {optimized_metrics['roc_auc']:.4f}")
print()

# Calculate improvements
print("Performance Improvements:")
for metric in baseline_metrics.keys():
    improvement = (optimized_metrics[metric] - baseline_metrics[metric]) * 100
    print(f"  {metric.capitalize():10s}: {improvement:+.2f}%")
print()

# ============================================================================
# 6. SAVE OPTIMIZED MODEL
# ============================================================================
print("Step 6: Saving optimized model...")
print("-" * 70)

joblib.dump(optimized_model, 'intrusion_detection_model_optimized.pkl')
print("✓ Optimized model saved as 'intrusion_detection_model_optimized.pkl'")
print()

# Save optimization results
results_df = pd.DataFrame({
    'Metric': list(baseline_metrics.keys()),
    'Baseline': list(baseline_metrics.values()),
    'Optimized': list(optimized_metrics.values()),
    'Improvement (%)': [(optimized_metrics[m] - baseline_metrics[m]) * 100 for m in baseline_metrics.keys()]
})
results_df.to_csv('optimization_results.csv', index=False)
print("✓ Results saved as 'optimization_results.csv'")
print()

# Save best parameters
best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv('best_hyperparameters.csv', index=False)
print("✓ Best parameters saved as 'best_hyperparameters.csv'")
print()

print("="*70)
print("OPTIMIZATION COMPLETE!")
print("="*70)
print("\nNext: Run 'python model_optimization_visualizations.py' to generate comparison charts")

