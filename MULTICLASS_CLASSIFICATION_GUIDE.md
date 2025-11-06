# Multi-class Classification Guide

This guide explains how to use multi-class classification to identify specific attack types in your Intrusion Detection System, going beyond simple binary (normal vs attack) classification.

## Overview

Multi-class classification enables your IDS to:
- **Identify specific attack types**: Know exactly what kind of attack is happening
- **Enable targeted responses**: Different attacks require different countermeasures
- **Provide actionable intelligence**: Security teams can prioritize based on attack type
- **Improve threat analysis**: Track trends and patterns for specific attack categories

## Attack Categories in UNSW-NB15

The dataset contains **10 classes** (1 normal + 9 attack types):

### 1. Normal
Legitimate network traffic with no malicious intent.

### 2. Fuzzers
Attempts to cause program or network suspension by feeding randomly generated data to find security vulnerabilities.

### 3. Analysis
Includes port scanning, spam, and HTML file penetrations. Attackers probe the network to gather information.

### 4. Backdoors
Techniques to bypass normal authentication and gain unauthorized access to systems.

### 5. DoS (Denial of Service)
Attacks that attempt to make network resources unavailable to legitimate users by overwhelming the system.

### 6. Exploits
Known exploits against vulnerable services, taking advantage of software bugs or weaknesses.

### 7. Generic
Techniques that work against all block ciphers without considering their structure.

### 8. Reconnaissance
Surveillance and probing attacks to gather information about the target network.

### 9. Shellcode
Exploits that execute shell commands on the target system, often used to gain control.

### 10. Worms
Self-replicating malware that spreads across networks without user intervention.

## Prerequisites

### 1. Original Dataset Required

Unlike binary classification, multi-class requires the **original UNSW-NB15 dataset** with the `attack_cat` column intact.

**Dataset files needed:**
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

**Download from:**
- [UNSW-NB15 Official Page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### 2. Update Data Paths

Edit the scripts to point to your dataset location:

```python
# In multiclass_data_exploration.py and multiclass_classification.py
data_paths = [
    "path/to/your/UNSW_NB15_training-set.csv",
    # Add your custom path here
]
```

### 3. Verify Setup

```bash
python3 test_setup.py
```

## Running Multi-class Classification

### Step 1: Explore Attack Distribution

First, understand the class distribution and imbalance:

```bash
python3 multiclass_data_exploration.py
```

**What it does:**
- Loads original UNSW-NB15 dataset
- Analyzes attack type distribution
- Calculates class imbalance ratio
- Generates distribution visualizations
- Provides recommendations for handling imbalance

**Expected output:**
```
MULTI-CLASS ATTACK TYPE ANALYSIS
================================================================

Attack Categories in Training Set:
  Normal              : 56000 (31.92%)
  Generic             : 40000 (22.81%)
  Exploits            : 33393 (19.04%)
  Fuzzers             : 18184 (10.37%)
  DoS                 : 12264 (6.99%)
  Reconnaissance      :  10491 (5.98%)
  Analysis            :   2000 (1.14%)
  Backdoors           :   1746 (1.00%)
  Shellcode           :   1133 (0.65%)
  Worms               :    130 (0.07%)

Class Imbalance Ratio: 430.77:1
```

**Generated files:**
- `multiclass_attack_distribution.png`
- `multiclass_attack_pie_chart.png`
- `multiclass_train_test_comparison.png`
- `multiclass_attack_summary.csv`

### Step 2: Train Multi-class Model

Train the Random Forest classifier for all 10 classes:

```bash
python3 multiclass_classification.py
```

**What it does:**
1. Loads original dataset with `attack_cat` column
2. Preprocesses data (encoding, normalization)
3. Encodes attack categories to numeric labels
4. Trains Random Forest with `class_weight='balanced'`
5. Evaluates on test set
6. Saves model and results

**Expected runtime:** 5-15 minutes (depends on dataset size and CPU)

**Key features:**
- **Class balancing**: Uses `class_weight='balanced'` to handle imbalanced classes
- **Optimized parameters**: Uses best parameters from hyperparameter tuning
- **Comprehensive metrics**: Per-class precision, recall, F1-score
- **Saves encoder**: Attack category encoder for making predictions

**Generated files:**
- `intrusion_detection_model_multiclass.pkl` - Trained model
- `attack_category_encoder.pkl` - Label encoder
- `train_multiclass_processed.csv` - Preprocessed training data
- `test_multiclass_processed.csv` - Preprocessed test data
- `multiclass_classification_report.csv` - Detailed metrics
- `multiclass_confusion_matrix.csv` - Confusion matrix
- `multiclass_feature_names.csv` - Feature list

### Step 3: Generate Visualizations

Create comprehensive visualizations of results:

```bash
python3 multiclass_visualizations.py
```

**What it does:**
- Loads trained model and test data
- Generates 6 different visualizations
- Saves feature importance analysis
- Creates confusion matrices (absolute and normalized)

**Expected runtime:** 2-5 minutes

**Generated files:**
1. `multiclass_confusion_matrix_heatmap.png`
2. `multiclass_confusion_matrix_normalized.png`
3. `multiclass_per_class_metrics.png`
4. `multiclass_f1_scores.png`
5. `multiclass_distribution_accuracy.png`
6. `multiclass_feature_importance.png`
7. `multiclass_feature_importance.csv`

## Understanding the Results

### Confusion Matrix

The confusion matrix shows how often each class is predicted correctly or confused with others.

**Reading the matrix:**
- **Diagonal values** (top-left to bottom-right): Correct predictions
- **Off-diagonal values**: Misclassifications
- **Row sums**: Total actual instances of each class
- **Column sums**: Total predicted instances of each class

**Common patterns:**
- **High diagonal values**: Good performance
- **Clusters off diagonal**: Certain attack types confused with each other
- **Sparse rows**: Rare attack types with few samples

### Per-Class Metrics

**Precision**: Of all predictions for this class, how many were correct?
- High precision = Few false positives
- Important for minimizing false alarms

**Recall**: Of all actual instances of this class, how many were detected?
- High recall = Few false negatives
- Important for catching all attacks

**F1-Score**: Harmonic mean of precision and recall
- Balances both metrics
- Good overall performance indicator

**Support**: Number of actual instances in test set
- Shows class distribution
- Affects metric reliability

### Typical Performance

Based on UNSW-NB15 characteristics:

**Well-detected classes (F1 > 0.80):**
- Normal traffic
- Generic attacks
- Exploits
- DoS

**Moderately detected (F1: 0.60-0.80):**
- Fuzzers
- Reconnaissance
- Analysis

**Challenging classes (F1 < 0.60):**
- Backdoors (rare, subtle)
- Shellcode (similar to exploits)
- Worms (very rare, only 130 samples)

### Class Imbalance Impact

**Symptoms:**
- High accuracy but poor recall for rare classes
- Model biased toward common classes
- Confusion between rare and common classes

**Solutions implemented:**
- `class_weight='balanced'` in Random Forest
- Stratified sampling in cross-validation
- Per-class metrics instead of just overall accuracy

## Making Predictions

### Using the Trained Model

```python
import pandas as pd
import joblib

# Load model and encoder
model = joblib.load('intrusion_detection_model_multiclass.pkl')
encoder = joblib.load('attack_category_encoder.pkl')

# Load and preprocess new data
# (must match training preprocessing)
new_data = pd.read_csv('new_network_traffic.csv')
# ... apply same preprocessing ...

# Make predictions
predictions = model.predict(new_data)

# Convert numeric labels to attack names
attack_names = encoder.inverse_transform(predictions)

# Show results
for i, attack in enumerate(attack_names):
    print(f"Sample {i}: {attack}")
```

### Prediction Confidence

```python
# Get prediction probabilities
probabilities = model.predict_proba(new_data)

# For each sample, show top 3 most likely classes
for i, probs in enumerate(probabilities):
    top_3_idx = probs.argsort()[-3:][::-1]
    print(f"\nSample {i}:")
    for idx in top_3_idx:
        class_name = encoder.classes_[idx]
        confidence = probs[idx] * 100
        print(f"  {class_name}: {confidence:.2f}%")
```

## Use Cases

### 1. Security Operations Center (SOC)

**Alert Prioritization:**
```
High Priority: DoS, Exploits, Backdoors
Medium Priority: Reconnaissance, Analysis
Low Priority: Fuzzers (if isolated)
```

**Automated Response:**
- DoS → Rate limiting, traffic filtering
- Backdoors → Isolate affected systems
- Reconnaissance → Increase monitoring
- Worms → Network segmentation

### 2. Threat Intelligence

**Track Attack Trends:**
- Which attack types are increasing?
- Correlation with external threat feeds
- Seasonal or time-based patterns
- Geographic attack type distribution

**Example Analysis:**
```python
# Count attacks by type over time
attack_counts = df.groupby(['date', 'attack_type']).size()
attack_counts.plot(kind='line', figsize=(12, 6))
```

### 3. Incident Response

**Faster Response:**
- Specific attack type → Specific playbook
- Pre-defined mitigation strategies
- Appropriate forensic tools
- Targeted threat hunting

**Example Workflow:**
```
1. IDS detects traffic
2. Multi-class model identifies: "Backdoor"
3. Trigger backdoor response playbook:
   - Isolate affected host
   - Capture memory dump
   - Review authentication logs
   - Check for persistence mechanisms
```

### 4. Compliance and Reporting

**Detailed Reporting:**
- Attack type breakdown for audits
- Regulatory compliance (PCI-DSS, HIPAA)
- Security posture assessment
- Risk quantification by attack type

## Troubleshooting

### Poor Performance on Rare Classes

**Problem:** Worms, Backdoors have very low F1-scores

**Solutions:**
1. **Collect more data** for rare classes
2. **Use SMOTE** (Synthetic Minority Over-sampling):
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```
3. **Adjust class weights** manually:
   ```python
   class_weights = {0: 1, 1: 1, ..., 9: 50}  # Higher weight for rare classes
   model = RandomForestClassifier(class_weight=class_weights)
   ```
4. **Use ensemble methods** with different sampling strategies

### Confusion Between Similar Attack Types

**Problem:** Shellcode confused with Exploits, Analysis with Reconnaissance

**Solutions:**
1. **Feature engineering**: Create attack-type-specific features
2. **Ensemble models**: Combine multiple classifiers
3. **Deep learning**: Neural networks can learn subtle differences
4. **Domain knowledge**: Add expert-defined rules for disambiguation

### Memory Errors

**Problem:** Out of memory during training

**Solutions:**
1. **Reduce n_estimators**: Use fewer trees (100 instead of 200)
2. **Limit max_depth**: Shallower trees use less memory
3. **Sample data**: Train on subset, validate on full set
4. **Use incremental learning**: Train in batches

### Long Training Time

**Problem:** Training takes too long

**Solutions:**
1. **Use n_jobs=-1**: Parallel processing
2. **Reduce parameter grid**: Fewer hyperparameter combinations
3. **Sample data**: Use stratified sampling for faster iteration
4. **Use RandomizedSearchCV**: Instead of GridSearchCV

## Advanced Usage

### Hierarchical Classification

First detect if traffic is attack, then classify attack type:

```python
# Step 1: Binary classification (fast)
is_attack = binary_model.predict(X)

# Step 2: Multi-class only for attacks (slower but more accurate)
attack_indices = np.where(is_attack == 1)[0]
if len(attack_indices) > 0:
    attack_types = multiclass_model.predict(X[attack_indices])
```

### Ensemble of Classifiers

Combine multiple models for better performance:

```python
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(...)),
        ('xgb', XGBClassifier(...)),
        ('lgb', LGBMClassifier(...))
    ],
    voting='soft'  # Use probabilities
)

ensemble.fit(X_train, y_train)
```

### Cost-Sensitive Learning

Assign different costs to different types of errors:

```python
# Define cost matrix (10x10 for 10 classes)
# cost[i][j] = cost of predicting j when true class is i
cost_matrix = np.ones((10, 10))
cost_matrix[0, :] = 10  # High cost for missing attacks (false negatives)
cost_matrix[:, 0] = 1   # Lower cost for false positives

# Implement in prediction
# (requires custom implementation or specialized libraries)
```

## Best Practices

1. **Always check class distribution** before training
2. **Use stratified splits** to maintain class proportions
3. **Monitor per-class metrics**, not just overall accuracy
4. **Validate on recent data** to detect concept drift
5. **Retrain periodically** as attack patterns evolve
6. **Document class mapping** for reproducibility
7. **Version control encoders** along with models
8. **Test on unseen attack variants** for robustness

## Next Steps

After multi-class classification:

1. **Deploy to Production**: Integrate with Snort or network monitoring
2. **Real-time Dashboard**: Visualize attack types in real-time
3. **Automated Response**: Trigger actions based on attack type
4. **Continuous Learning**: Retrain with new attack samples
5. **Deep Learning**: Try LSTM or CNN for better performance

## References

- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Scikit-learn Multi-class Classification](https://scikit-learn.org/stable/modules/multiclass.html)
- [Handling Imbalanced Classes](https://imbalanced-learn.org/)
- [Random Forest for Multi-class](https://scikit-learn.org/stable/modules/ensemble.html#forest)

## Support

For issues or questions:
1. Check this guide first
2. Review console output for errors
3. Verify dataset paths are correct
4. Ensure original UNSW-NB15 dataset is used

---

**Happy Classifying! 🎯**

