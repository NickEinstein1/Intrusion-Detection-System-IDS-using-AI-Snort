"""
Visualization Script for Multi-class Classification Results
Generates comprehensive visualizations for multi-class intrusion detection.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("GENERATING MULTI-CLASS VISUALIZATIONS")
print("="*70)
print()

# ============================================================================
# LOAD DATA AND MODEL
# ============================================================================
print("Loading data and model...")

try:
    # Load preprocessed data
    test_data = pd.read_csv('test_multiclass_processed.csv')
    
    # Load model and encoder
    model = joblib.load('intrusion_detection_model_multiclass.pkl')
    attack_encoder = joblib.load('attack_category_encoder.pkl')
    
    # Prepare features and labels
    X_test = test_data.drop(['attack_cat', 'attack_cat_encoded'], axis=1)
    y_test = test_data['attack_cat_encoded']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    print("✓ All files loaded successfully")
    print(f"✓ Number of classes: {len(attack_encoder.classes_)}")
    print()
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please run 'multiclass_classification.py' first.")
    exit(1)

# ============================================================================
# 1. CONFUSION MATRIX HEATMAP
# ============================================================================
print("Creating confusion matrix heatmap...")

cm = confusion_matrix(y_test, y_pred)
class_names = attack_encoder.classes_

fig, ax = plt.subplots(figsize=(14, 12))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Samples'}, linewidths=0.5)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Multi-class Confusion Matrix\nIntrusion Detection System', 
             fontsize=14, fontweight='bold', pad=20)

# Rotate labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('multiclass_confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_confusion_matrix_heatmap.png")
plt.close()

# ============================================================================
# 2. NORMALIZED CONFUSION MATRIX
# ============================================================================
print("Creating normalized confusion matrix...")

# Normalize by true labels (rows)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'}, linewidths=0.5, vmin=0, vmax=1)

ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Normalized Confusion Matrix (by True Label)\nIntrusion Detection System', 
             fontsize=14, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('multiclass_confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_confusion_matrix_normalized.png")
plt.close()

# ============================================================================
# 3. PER-CLASS PERFORMANCE METRICS
# ============================================================================
print("Creating per-class performance chart...")

# Get classification report as dictionary
report_dict = classification_report(y_test, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)

# Extract per-class metrics
classes = class_names
precision_scores = [report_dict[cls]['precision'] for cls in classes]
recall_scores = [report_dict[cls]['recall'] for cls in classes]
f1_scores = [report_dict[cls]['f1-score'] for cls in classes]

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(classes))
width = 0.25

bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)

ax.set_xlabel('Attack Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('multiclass_per_class_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_per_class_metrics.png")
plt.close()

# ============================================================================
# 4. F1-SCORE COMPARISON
# ============================================================================
print("Creating F1-score comparison chart...")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by F1-score
sorted_indices = np.argsort(f1_scores)
sorted_classes = [classes[i] for i in sorted_indices]
sorted_f1 = [f1_scores[i] for i in sorted_indices]

colors = ['red' if score < 0.5 else 'orange' if score < 0.7 else 'green' for score in sorted_f1]

bars = ax.barh(range(len(sorted_classes)), sorted_f1, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(sorted_classes)))
ax.set_yticklabels(sorted_classes)
ax.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Attack Category', fontsize=12, fontweight='bold')
ax.set_title('F1-Score by Attack Category (Sorted)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim([0, 1.1])
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
    ax.text(score, i, f' {score:.3f}', va='center', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, label='Good (≥0.70)'),
    Patch(facecolor='orange', alpha=0.7, label='Fair (0.50-0.69)'),
    Patch(facecolor='red', alpha=0.7, label='Poor (<0.50)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('multiclass_f1_scores.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_f1_scores.png")
plt.close()

# ============================================================================
# 5. CLASS DISTRIBUTION AND ACCURACY
# ============================================================================
print("Creating class distribution and accuracy chart...")

# Calculate per-class accuracy
class_accuracies = []
class_counts = []

for i, class_name in enumerate(class_names):
    mask = (y_test == i)
    if mask.sum() > 0:
        class_acc = (y_pred[mask] == y_test[mask]).sum() / mask.sum()
        class_accuracies.append(class_acc)
        class_counts.append(mask.sum())
    else:
        class_accuracies.append(0)
        class_counts.append(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top: Class distribution
ax1.bar(range(len(class_names)), class_counts, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(class_names)))
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
ax1.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, count in enumerate(class_counts):
    ax1.text(i, count, f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Bottom: Per-class accuracy
colors_acc = ['green' if acc >= 0.8 else 'orange' if acc >= 0.6 else 'red' for acc in class_accuracies]
ax2.bar(range(len(class_names)), class_accuracies, color=colors_acc, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(class_names)))
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.set_xlabel('Attack Category', fontsize=11, fontweight='bold')
ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
ax2.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.1])
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, acc in enumerate(class_accuracies):
    if acc > 0:
        ax2.text(i, acc, f'{acc:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('multiclass_distribution_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_distribution_accuracy.png")
plt.close()

# ============================================================================
# 6. FEATURE IMPORTANCE FOR MULTI-CLASS
# ============================================================================
print("Creating feature importance chart...")

# Get feature importances
importances = model.feature_importances_
feature_names = pd.read_csv('multiclass_feature_names.csv')['feature'].tolist()

# Create DataFrame and sort
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 20 features
fig, ax = plt.subplots(figsize=(12, 8))
top_20 = feature_importance_df.head(20)

bars = ax.barh(range(len(top_20)), top_20['Importance'], color='coral', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Feature'])
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Top 20 Most Important Features\nMulti-class Classification', 
             fontsize=14, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, importance) in enumerate(zip(bars, top_20['Importance'])):
    ax.text(importance, i, f' {importance:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('multiclass_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_feature_importance.png")
plt.close()

# Save full feature importance
feature_importance_df.to_csv('multiclass_feature_importance.csv', index=False)
print("✓ Saved: multiclass_feature_importance.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*70)
print("VISUALIZATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. multiclass_confusion_matrix_heatmap.png")
print("  2. multiclass_confusion_matrix_normalized.png")
print("  3. multiclass_per_class_metrics.png")
print("  4. multiclass_f1_scores.png")
print("  5. multiclass_distribution_accuracy.png")
print("  6. multiclass_feature_importance.png")
print("  7. multiclass_feature_importance.csv")
print()

