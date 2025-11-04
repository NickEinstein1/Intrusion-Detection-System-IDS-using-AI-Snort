# Feature Importance Analysis Guide

This guide explains how to use the feature importance analysis tools to understand which network features are most critical for detecting intrusions.

## Overview

Feature importance analysis helps you:
- **Identify critical features**: Understand which network traffic characteristics are most indicative of attacks
- **Optimize the model**: Reduce dimensionality by focusing on the most important features
- **Gain insights**: Learn about attack patterns and network behavior
- **Improve performance**: Potentially speed up predictions by using fewer features

## Prerequisites

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

Or verify your setup:
```bash
python3 test_setup.py
```

### 2. Generate Preprocessed Data

If you haven't already, you need to run the main notebook to generate the preprocessed training data:

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook Intrusion_Detection_System(IDS).ipynb
   ```

2. Run all cells up to and including the "Model Training" section

3. This will create:
   - `train_processed.csv` - Preprocessed training data
   - `test_processed.csv` - Preprocessed test data
   - `intrusion_detection_model_unsw.pkl` - Trained model

## Running the Analysis

### Method 1: Jupyter Notebook (Recommended)

1. Open the notebook:
   ```bash
   jupyter notebook Intrusion_Detection_System(IDS).ipynb
   ```

2. Navigate to the **"Feature Importance Analysis"** section (near the end)

3. Run all cells in that section

4. View the visualizations inline and interact with the data

### Method 2: Python Script

Run the standalone script:
```bash
python3 feature_importance_analysis.py
```

This will:
- Load the trained model and preprocessed data
- Extract feature importances
- Generate all visualizations
- Save results to files
- Print summary statistics

## Generated Outputs

### Visualizations

1. **feature_importance_top20.png**
   - Horizontal bar chart of the 20 most important features
   - Easy to read and compare feature importance scores
   - Best for presentations and reports

2. **feature_importance_top15_vertical.png**
   - Vertical bar chart with exact importance values
   - Shows the top 15 features with numerical labels
   - Useful for detailed analysis

3. **cumulative_feature_importance.png**
   - Line plot showing cumulative importance
   - Indicates how many features capture 90% and 95% of predictive power
   - Helps determine optimal feature subset size

4. **top10_features_correlation.png**
   - Heatmap showing correlations between top 10 features
   - Identifies redundant or complementary features
   - Useful for feature engineering

### Data Files

1. **feature_importance_full.csv**
   - Complete ranking of all features with importance scores
   - Two columns: Feature name and Importance score
   - Sorted by importance (highest to lowest)
   - Can be used for further analysis in Excel or other tools

## Understanding the Results

### Feature Importance Scores

- **Range**: 0.0 to 1.0 (all scores sum to 1.0)
- **Interpretation**: Higher score = more important for predictions
- **Example**: A score of 0.15 means the feature contributes 15% to the model's decisions

### Key Metrics

The analysis provides several insights:

1. **Top Features**: Which features are most critical
2. **Cumulative Importance**: How many features you really need
3. **Feature Correlations**: Which features are related
4. **Distribution**: How importance is spread across features

### Typical Findings

Based on network intrusion detection, you might find:
- **Flow-based features** (duration, bytes, packets) are often highly important
- **TCP connection features** (window size, TTL) can be critical
- **Rate-based features** (packets per second) help detect anomalies
- **Service and protocol** information provides context

## Use Cases

### 1. Model Optimization

If the analysis shows that 20 features capture 95% of importance:
- Retrain the model using only those 20 features
- Reduce computational cost
- Potentially improve generalization

### 2. Feature Engineering

If highly correlated features are both important:
- Consider creating combined features
- Remove redundant features
- Engineer new features based on relationships

### 3. Domain Understanding

Use the results to:
- Validate that important features make sense for intrusion detection
- Identify unexpected patterns
- Guide data collection priorities

### 4. Reporting and Communication

Use the visualizations to:
- Explain the model to stakeholders
- Justify feature selection decisions
- Document model behavior

## Troubleshooting

### Error: "File not found: train_processed.csv"

**Solution**: Run the main notebook first to generate preprocessed data.

### Error: "Module not found"

**Solution**: Install required packages:
```bash
pip install -r requirements.txt
```

### Visualizations not displaying in Jupyter

**Solution**: Add this at the top of the notebook:
```python
%matplotlib inline
```

### Script runs but no output files

**Solution**: Check write permissions in the current directory.

## Advanced Usage

### Analyzing Specific Feature Subsets

Modify the script to analyze specific features:

```python
# In feature_importance_analysis.py or notebook
specific_features = ['sbytes', 'dbytes', 'rate', 'sttl', 'dttl']
subset_data = train_data[specific_features]
# Analyze correlations, distributions, etc.
```

### Comparing Multiple Models

If you train different models, compare their feature importances:

```python
model1 = joblib.load('model1.pkl')
model2 = joblib.load('model2.pkl')

importance_comparison = pd.DataFrame({
    'Feature': feature_names,
    'Model1': model1.feature_importances_,
    'Model2': model2.feature_importances_
})
```

### Exporting for External Tools

The CSV file can be imported into:
- **Excel**: For custom charts and analysis
- **Tableau/Power BI**: For interactive dashboards
- **R**: For statistical analysis
- **Python notebooks**: For further exploration

## Best Practices

1. **Run after model training**: Always generate fresh importance scores after retraining
2. **Compare across datasets**: Check if importance is consistent across different data splits
3. **Validate findings**: Ensure important features make domain sense
4. **Document insights**: Keep notes on what you learn from the analysis
5. **Version control**: Save importance scores with model versions

## Next Steps

After analyzing feature importance, consider:

1. **Cross-validation**: Verify importance scores are stable across folds
2. **Hyperparameter tuning**: Optimize model with important features
3. **Feature selection**: Retrain with reduced feature set
4. **Multi-class classification**: Analyze importance for specific attack types
5. **Real-time deployment**: Use insights to optimize production systems

## References

- [Scikit-learn Feature Importance](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)
- [Random Forest Feature Selection](https://towardsdatascience.com/feature-selection-using-random-forest-26d7b747597f)
- [UNSW-NB15 Dataset Documentation](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)

## Support

For issues or questions:
1. Check this guide first
2. Review the main README.md
3. Open an issue on GitHub
4. Check the Jupyter notebook comments

---

**Happy Analyzing! 📊**

