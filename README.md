# Intrusion Detection System (IDS) using AI & Snort

![Intrusion Detection System](docs/Intrusion_detection.jpg)

An AI-powered Intrusion Detection System that uses machine learning to detect network intrusions and attacks. This project implements a Random Forest classifier trained on the UNSW-NB15 dataset to identify malicious network traffic with high accuracy.

## Overview

This project demonstrates the application of machine learning techniques for cybersecurity, specifically for detecting network intrusions. The system achieves **90% accuracy** in distinguishing between normal and malicious network traffic.

## Features

- **Machine Learning-Based Detection**: Uses Random Forest classifier for robust intrusion detection
- **High Accuracy**: Achieves 90% accuracy on the UNSW-NB15 test dataset
- **Comprehensive Data Preprocessing**: Handles categorical encoding, normalization, and missing values
- **Binary Classification**: Classifies network traffic as normal (0) or attack (1)
- **Trained Model Export**: Saves the trained model for deployment and reuse

## Dataset

The project uses the **UNSW-NB15 dataset**, which contains:
- Modern network traffic patterns
- Multiple attack categories
- 45 features including:
  - Network flow features (duration, protocol, service, state)
  - Packet statistics (spkts, dpkts, sbytes, dbytes)
  - TCP connection features (sttl, dttl, swin, dwin)
  - Time-based features (rate, sload, dload)
  - And many more network-specific attributes

## Technologies Used

- **Python 3.13**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and preprocessing
- **Jupyter Notebook**: Interactive development environment

## Requirements

```
pandas
scikit-learn
numpy
joblib
matplotlib
seaborn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Intrusion-Detection-System-IDS-using-AI-Snort.git
cd Intrusion-Detection-System-IDS-using-AI-Snort
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn
```

## Usage

### Training the Model

1. Open the Jupyter Notebook:
```bash
jupyter notebook Intrusion_Detection_System(IDS).ipynb
```

2. The notebook includes the following steps:
   - **Data Loading**: Load UNSW-NB15 training and testing datasets
   - **Data Preprocessing**: 
     - Drop unnecessary columns (id, attack_cat)
     - Handle missing values
     - Encode categorical features
     - Normalize numerical features
   - **Model Training**: Train Random Forest classifier with 100 estimators
   - **Model Evaluation**: Evaluate performance on test data
   - **Model Export**: Save trained model as pickle file

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('intrusion_detection_model_unsw.pkl')

# Prepare your data (must be preprocessed in the same way as training data)
# X_new = your_preprocessed_data

# Make predictions
predictions = model.predict(X_new)
# 0 = Normal traffic, 1 = Attack
```

## Model Performance

The Random Forest classifier achieves the following performance metrics:

- **Overall Accuracy**: 90.09%

### Classification Report:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal (0) | 0.77 | 0.98 | 0.86 | 56,000 |
| Attack (1) | 0.99 | 0.87 | 0.92 | 119,341 |
| **Weighted Avg** | **0.92** | **0.90** | **0.90** | **175,341** |

### Key Insights:
- **High Precision for Attacks (0.99)**: Very few false positives when detecting attacks
- **High Recall for Normal Traffic (0.98)**: Excellent at identifying legitimate traffic
- **Balanced Performance**: Good balance between detecting attacks and minimizing false alarms

## Project Structure

```
Intrusion-Detection-System-IDS-using-AI-Snort/
├── Intrusion_Detection_System(IDS).ipynb  # Main Jupyter notebook with training & analysis
├── feature_importance_analysis.py         # Standalone script for feature analysis
├── model_optimization.py                  # Cross-validation and hyperparameter tuning
├── model_optimization_visualizations.py   # Optimization results visualizations
├── test_setup.py                          # Setup verification script
├── intrusion_detection_model_unsw.pkl     # Trained baseline Random Forest model
├── intrusion_detection_model_optimized.pkl # Optimized model (generated)
├── feature_importance_full.csv            # Complete feature importance rankings
├── optimization_results.csv               # Model optimization metrics (generated)
├── best_hyperparameters.csv               # Optimal hyperparameters (generated)
├── requirements.txt                       # Python package dependencies
├── *.png                                  # Generated visualization charts
├── docs/                                  # Documentation and images
│   └── Intrusion_detection.jpg            # Project diagram
├── README.md                              # This file
├── FEATURE_IMPORTANCE_GUIDE.md            # Detailed guide for feature analysis
├── MODEL_OPTIMIZATION_GUIDE.md            # Detailed guide for model optimization
├── LICENSE                                # MIT License
├── pyvenv.cfg                             # Virtual environment config
├── .gitignore                             # Git ignore rules
└── .gitattributes                         # Git LFS configuration
```

## Workflow

1. **Data Preprocessing**
   - Load UNSW-NB15 training and testing datasets
   - Remove unnecessary columns
   - Handle missing values
   - Encode categorical variables using LabelEncoder
   - Normalize numerical features using StandardScaler

2. **Model Training**
   - Initialize Random Forest Classifier (100 estimators)
   - Train on preprocessed training data
   - Random state set to 42 for reproducibility

3. **Model Evaluation**
   - Test on preprocessed test data
   - Generate accuracy metrics and classification report

4. **Model Deployment**
   - Export trained model using joblib
   - Model ready for integration with network monitoring systems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Nick**

## Acknowledgments

- UNSW-NB15 dataset creators for providing a comprehensive modern network traffic dataset
- scikit-learn community for excellent machine learning tools
- Open-source community for continuous support and inspiration

## Feature Importance Analysis

Understanding which network features are most critical for detecting intrusions helps optimize the model and provides insights into attack patterns.

### Quick Start

**Verify Setup:**
```bash
python3 test_setup.py
```

**Run Analysis:**
```bash
# Option 1: Python Script
python3 feature_importance_analysis.py

# Option 2: Jupyter Notebook
jupyter notebook Intrusion_Detection_System(IDS).ipynb
# Navigate to the "Feature Importance Analysis" section
```

### Generated Outputs

   #### Visualizations

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
   
   #### Data Files

   1. **feature_importance_full.csv**
      - Complete ranking of all features with importance scores
      - Two columns: Feature name and Importance score
      - Sorted by importance (highest to lowest)
      - Can be used for further analysis in Excel or other tools

### Key Insights

The feature importance analysis reveals:
- Which network traffic characteristics are most indicative of attacks
- How many features are truly necessary for accurate detection
- Correlations between important features
- Opportunities for feature engineering and model optimization

**For detailed instructions, see [FEATURE_IMPORTANCE_GUIDE.md](FEATURE_IMPORTANCE_GUIDE.md)**

## Model Optimization

Improve model performance through cross-validation and hyperparameter tuning.

### Quick Start

**Run Optimization:**
```bash
# Step 1: Perform cross-validation and hyperparameter tuning
python3 model_optimization.py

# Step 2: Generate comparison visualizations
python3 model_optimization_visualizations.py
```

### What It Does

1. **K-Fold Cross-Validation**
   - Evaluates model stability across 5 different data splits
   - Provides mean accuracy with confidence intervals
   - Identifies overfitting or underfitting issues

2. **Grid Search Hyperparameter Tuning**
   - Tests multiple parameter combinations
   - Finds optimal settings for:
     - Number of trees (`n_estimators`)
     - Tree depth (`max_depth`)
     - Split criteria (`min_samples_split`, `min_samples_leaf`)
     - Feature selection (`max_features`)
     - Bootstrap sampling (`bootstrap`)

3. **Performance Comparison**
   - Compares baseline vs optimized model
   - Calculates improvement percentages
   - Generates detailed visualizations

### Generated Outputs

#### Optimized Model
- **intrusion_detection_model_optimized.pkl** - Tuned Random Forest model with best parameters

#### Performance Data
- **optimization_results.csv** - Detailed metrics comparison (accuracy, precision, recall, F1, ROC-AUC)
- **best_hyperparameters.csv** - Optimal parameter values found by GridSearch

#### Visualizations

1. **optimization_metrics_comparison.png**
   - Side-by-side bar chart comparing all metrics
   - Shows baseline vs optimized performance

2. **optimization_improvement.png**
   - Horizontal bar chart showing percentage improvements
   - Highlights which metrics improved most

3. **optimization_confusion_matrices.png**
   - Confusion matrices for both models
   - Visual comparison of prediction accuracy

4. **optimization_roc_curves.png**
   - ROC curves with AUC scores
   - Demonstrates improved classification ability

5. **optimization_learning_curves.png**
   - Training vs validation scores across dataset sizes
   - Identifies overfitting/underfitting patterns

### Key Benefits

- **Better Accuracy**: Optimized hyperparameters improve detection rates
- **Reduced False Positives**: Better precision means fewer false alarms
- **Model Stability**: Cross-validation ensures consistent performance
- **Data-Driven Decisions**: Systematic parameter search vs manual tuning
- **Production Ready**: Validated model ready for deployment

**For detailed instructions, see [MODEL_OPTIMIZATION_GUIDE.md](MODEL_OPTIMIZATION_GUIDE.md)**

## Future Enhancements

- [x] Feature importance analysis and visualization
- [x] Cross-validation and hyperparameter tuning
- [ ] Integration with Snort IDS for real-time detection
- [ ] Multi-class classification for specific attack types
- [ ] Deep learning models (LSTM, CNN) for improved accuracy
- [ ] Real-time network traffic monitoring dashboard
- [ ] Deployment as a REST API service
- [ ] Docker containerization for easy deployment

## Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Note**: This project is for educational and research purposes. Always ensure proper authorization before deploying intrusion detection systems on any network.