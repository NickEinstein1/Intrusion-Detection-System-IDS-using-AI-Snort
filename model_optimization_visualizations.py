"""
Visualization Script for Model Optimization Results where we analyze the results of the model optimization process.
Generates comparison charts between baseline and optimized models.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("GENERATING MODEL OPTIMIZATION VISUALIZATIONS")
print("="*70)
print()

# ============================================================================
# LOAD DATA AND MODELS
# ============================================================================
print("Loading data and models...")

try:
    # Load data
    train_data = pd.read_csv('train_processed.csv')
    test_data = pd.read_csv('test_processed.csv')
    
    X_train = train_data.drop('attack_cat', axis=1)
    y_train = train_data['attack_cat']
    X_test = test_data.drop('attack_cat', axis=1)
    y_test = test_data['attack_cat']
    
    # Load models
    baseline_model = joblib.load('intrusion_detection_model_unsw.pkl')
    optimized_model = joblib.load('intrusion_detection_model_optimized.pkl')
    
    # Load results
    results_df = pd.read_csv('optimization_results.csv')
    
    print("✓ All files loaded successfully")
    print()
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run 'model_optimization.py' first.")
    exit(1)

# ============================================================================
# 1. METRICS COMPARISON BAR CHART
# ============================================================================
print("Creating metrics comparison chart...")

fig, ax = plt.subplots(figsize=(12, 6))

metrics = results_df['Metric'].values
baseline_values = results_df['Baseline'].values
optimized_values = results_df['Optimized'].values

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline Model', alpha=0.8)
bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized Model', alpha=0.8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Comparison: Baseline vs Optimized', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: optimization_metrics_comparison.png")
plt.close()

# ============================================================================
# 2. IMPROVEMENT PERCENTAGE CHART
# ============================================================================
print("Creating improvement percentage chart...")

fig, ax = plt.subplots(figsize=(10, 6))

improvements = results_df['Improvement (%)'].values
colors = ['green' if x > 0 else 'red' for x in improvements]

bars = ax.barh(metrics, improvements, color=colors, alpha=0.7)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    ax.text(val, i, f' {val:+.2f}%', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Metrics', fontsize=12, fontweight='bold')
ax.set_title('Performance Improvement After Optimization', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_yticklabels([m.capitalize() for m in metrics])
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_improvement.png', dpi=300, bbox_inches='tight')
print("✓ Saved: optimization_improvement.png")
plt.close()

# ============================================================================
# 3. CONFUSION MATRICES COMPARISON
# ============================================================================
print("Creating confusion matrices comparison...")

baseline_pred = baseline_model.predict(X_test)
optimized_pred = optimized_model.predict(X_test)

cm_baseline = confusion_matrix(y_test, baseline_pred)
cm_optimized = confusion_matrix(y_test, optimized_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline confusion matrix
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            cbar_kws={'label': 'Count'})
axes[0].set_title('Baseline Model\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=10)
axes[0].set_ylabel('True Label', fontsize=10)
axes[0].set_xticklabels(['Normal', 'Attack'])
axes[0].set_yticklabels(['Normal', 'Attack'])

# Optimized confusion matrix
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            cbar_kws={'label': 'Count'})
axes[1].set_title('Optimized Model\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Predicted Label', fontsize=10)
axes[1].set_ylabel('True Label', fontsize=10)
axes[1].set_xticklabels(['Normal', 'Attack'])
axes[1].set_yticklabels(['Normal', 'Attack'])

plt.tight_layout()
plt.savefig('optimization_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Saved: optimization_confusion_matrices.png")
plt.close()

# ============================================================================
# 4. ROC CURVES COMPARISON
# ============================================================================
print("Creating ROC curves comparison...")

baseline_proba = baseline_model.predict_proba(X_test)[:, 1]
optimized_proba = optimized_model.predict_proba(X_test)[:, 1]

fpr_baseline, tpr_baseline, _ = roc_curve(y_test, baseline_proba)
fpr_optimized, tpr_optimized, _ = roc_curve(y_test, optimized_proba)

auc_baseline = roc_auc_score(y_test, baseline_proba)
auc_optimized = roc_auc_score(y_test, optimized_proba)

fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr_baseline, tpr_baseline, label=f'Baseline Model (AUC = {auc_baseline:.4f})', 
        linewidth=2, alpha=0.8)
ax.plot(fpr_optimized, tpr_optimized, label=f'Optimized Model (AUC = {auc_optimized:.4f})', 
        linewidth=2, alpha=0.8)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_roc_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: optimization_roc_curves.png")
plt.close()

# ============================================================================
# 5. LEARNING CURVES
# ============================================================================
print("Creating learning curves (this may take a moment)...")

train_sizes = np.linspace(0.1, 1.0, 10)

# Baseline learning curve
train_sizes_baseline, train_scores_baseline, val_scores_baseline = learning_curve(
    baseline_model, X_train, y_train, train_sizes=train_sizes, cv=3, 
    scoring='accuracy', n_jobs=-1, random_state=42
)

# Optimized learning curve
train_sizes_optimized, train_scores_optimized, val_scores_optimized = learning_curve(
    optimized_model, X_train, y_train, train_sizes=train_sizes, cv=3,
    scoring='accuracy', n_jobs=-1, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Baseline learning curve
axes[0].plot(train_sizes_baseline, train_scores_baseline.mean(axis=1), 
             label='Training Score', marker='o', linewidth=2)
axes[0].plot(train_sizes_baseline, val_scores_baseline.mean(axis=1), 
             label='Validation Score', marker='s', linewidth=2)
axes[0].fill_between(train_sizes_baseline, 
                      train_scores_baseline.mean(axis=1) - train_scores_baseline.std(axis=1),
                      train_scores_baseline.mean(axis=1) + train_scores_baseline.std(axis=1),
                      alpha=0.2)
axes[0].fill_between(train_sizes_baseline,
                      val_scores_baseline.mean(axis=1) - val_scores_baseline.std(axis=1),
                      val_scores_baseline.mean(axis=1) + val_scores_baseline.std(axis=1),
                      alpha=0.2)
axes[0].set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Accuracy Score', fontsize=11, fontweight='bold')
axes[0].set_title('Baseline Model Learning Curve', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# Optimized learning curve
axes[1].plot(train_sizes_optimized, train_scores_optimized.mean(axis=1),
             label='Training Score', marker='o', linewidth=2)
axes[1].plot(train_sizes_optimized, val_scores_optimized.mean(axis=1),
             label='Validation Score', marker='s', linewidth=2)
axes[1].fill_between(train_sizes_optimized,
                      train_scores_optimized.mean(axis=1) - train_scores_optimized.std(axis=1),
                      train_scores_optimized.mean(axis=1) + train_scores_optimized.std(axis=1),
                      alpha=0.2)
axes[1].fill_between(train_sizes_optimized,
                      val_scores_optimized.mean(axis=1) - val_scores_optimized.std(axis=1),
                      val_scores_optimized.mean(axis=1) + val_scores_optimized.std(axis=1),
                      alpha=0.2)
axes[1].set_xlabel('Training Set Size', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Accuracy Score', fontsize=11, fontweight='bold')
axes[1].set_title('Optimized Model Learning Curve', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_learning_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: optimization_learning_curves.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. optimization_metrics_comparison.png")
print("  2. optimization_improvement.png")
print("  3. optimization_confusion_matrices.png")
print("  4. optimization_roc_curves.png")
print("  5. optimization_learning_curves.png")
print()

