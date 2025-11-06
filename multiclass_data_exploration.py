"""
Multi-class Data Exploration Script
Analyzes the UNSW-NB15 dataset to understand attack type distribution.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("MULTI-CLASS ATTACK TYPE ANALYSIS")
print("="*70)
print()

# ============================================================================
# 1. LOAD ORIGINAL DATA
# ============================================================================
print("Step 1: Loading original UNSW-NB15 dataset...")
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
# 2. ANALYZE ATTACK CATEGORIES
# ============================================================================
print("Step 2: Analyzing attack categories...")
print("-" * 70)

# Check if attack_cat column exists
if 'attack_cat' not in train_data.columns:
    print("✗ Error: 'attack_cat' column not found in dataset")
    print(f"Available columns: {train_data.columns.tolist()}")
    exit(1)

# Get unique attack categories
attack_categories_train = train_data['attack_cat'].value_counts()
attack_categories_test = test_data['attack_cat'].value_counts()

print("\nAttack Categories in Training Set:")
print("=" * 50)
for category, count in attack_categories_train.items():
    percentage = (count / len(train_data)) * 100
    print(f"  {category:20s}: {count:6d} ({percentage:5.2f}%)")

print("\nAttack Categories in Test Set:")
print("=" * 50)
for category, count in attack_categories_test.items():
    percentage = (count / len(test_data)) * 100
    print(f"  {category:20s}: {count:6d} ({percentage:5.2f}%)")

print()

# ============================================================================
# 3. CLASS IMBALANCE ANALYSIS
# ============================================================================
print("Step 3: Class imbalance analysis...")
print("-" * 70)

# Combine train and test for overall statistics
combined_data = pd.concat([train_data, test_data], ignore_index=True)
attack_distribution = combined_data['attack_cat'].value_counts()

print("\nOverall Attack Distribution (Train + Test):")
print("=" * 50)
print(f"Total samples: {len(combined_data)}")
print(f"Number of classes: {len(attack_distribution)}")
print()

for category, count in attack_distribution.items():
    percentage = (count / len(combined_data)) * 100
    print(f"  {category:20s}: {count:7d} ({percentage:5.2f}%)")

# Calculate imbalance ratio
max_class = attack_distribution.max()
min_class = attack_distribution.min()
imbalance_ratio = max_class / min_class

print()
print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"  (Largest class / Smallest class)")
print()

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================
print("Step 4: Creating visualizations...")
print("-" * 70)

# Visualization 1: Attack distribution bar chart
fig, ax = plt.subplots(figsize=(14, 8))
categories = attack_distribution.index.tolist()
counts = attack_distribution.values

bars = ax.bar(range(len(categories)), counts, color='steelblue', edgecolor='black', linewidth=1.2)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.set_xlabel('Attack Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_title('Attack Type Distribution in UNSW-NB15 Dataset', fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    percentage = (count / len(combined_data)) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('multiclass_attack_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_attack_distribution.png")
plt.close()

# Visualization 2: Pie chart
fig, ax = plt.subplots(figsize=(12, 10))
colors = sns.color_palette("husl", len(categories))
wedges, texts, autotexts = ax.pie(counts, labels=categories, autopct='%1.1f%%',
                                    colors=colors, startangle=90,
                                    textprops={'fontsize': 10, 'fontweight': 'bold'})
ax.set_title('Attack Type Distribution (Percentage)', fontsize=14, fontweight='bold', pad=20)

# Make percentage text more readable
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)

plt.tight_layout()
plt.savefig('multiclass_attack_pie_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_attack_pie_chart.png")
plt.close()

# Visualization 3: Train vs Test distribution comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training set
train_categories = attack_categories_train.index.tolist()
train_counts = attack_categories_train.values
axes[0].bar(range(len(train_categories)), train_counts, color='coral', edgecolor='black', linewidth=1.2)
axes[0].set_xticks(range(len(train_categories)))
axes[0].set_xticklabels(train_categories, rotation=45, ha='right')
axes[0].set_xlabel('Attack Category', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Test set
test_categories = attack_categories_test.index.tolist()
test_counts = attack_categories_test.values
axes[1].bar(range(len(test_categories)), test_counts, color='lightgreen', edgecolor='black', linewidth=1.2)
axes[1].set_xticks(range(len(test_categories)))
axes[1].set_xticklabels(test_categories, rotation=45, ha='right')
axes[1].set_xlabel('Attack Category', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('multiclass_train_test_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: multiclass_train_test_comparison.png")
plt.close()

# ============================================================================
# 5. SAVE ANALYSIS RESULTS
# ============================================================================
print()
print("Step 5: Saving analysis results...")
print("-" * 70)

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Attack_Category': attack_distribution.index,
    'Total_Count': attack_distribution.values,
    'Percentage': (attack_distribution.values / len(combined_data)) * 100,
    'Train_Count': [attack_categories_train.get(cat, 0) for cat in attack_distribution.index],
    'Test_Count': [attack_categories_test.get(cat, 0) for cat in attack_distribution.index]
})

summary_df.to_csv('multiclass_attack_summary.csv', index=False)
print("✓ Saved: multiclass_attack_summary.csv")
print()

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("="*70)
print("ANALYSIS COMPLETE - RECOMMENDATIONS")
print("="*70)
print()

print("Key Findings:")
print(f"  • Total attack categories: {len(attack_distribution)}")
print(f"  • Class imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"  • Most common category: {attack_distribution.index[0]} ({attack_distribution.values[0]} samples)")
print(f"  • Least common category: {attack_distribution.index[-1]} ({attack_distribution.values[-1]} samples)")
print()

print("Recommendations for Multi-class Classification:")
print()

if imbalance_ratio > 10:
    print("  ⚠ HIGH CLASS IMBALANCE DETECTED")
    print("  Recommended strategies:")
    print("    1. Use class_weight='balanced' in Random Forest")
    print("    2. Apply SMOTE (Synthetic Minority Over-sampling)")
    print("    3. Use stratified sampling for train/test splits")
    print("    4. Consider ensemble methods with resampling")
    print()
elif imbalance_ratio > 5:
    print("  ⚠ MODERATE CLASS IMBALANCE")
    print("  Recommended strategies:")
    print("    1. Use class_weight='balanced' in Random Forest")
    print("    2. Use stratified cross-validation")
    print()
else:
    print("  ✓ RELATIVELY BALANCED CLASSES")
    print("  Standard multi-class classification should work well")
    print()

print("Next Steps:")
print("  1. Run 'python multiclass_classification.py' to train multi-class model")
print("  2. Review generated visualizations")
print("  3. Compare multi-class vs binary classification performance")
print()

