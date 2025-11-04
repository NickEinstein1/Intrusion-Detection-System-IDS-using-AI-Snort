"""
Feature Importance Analysis for Intrusion Detection System
This script analyzes and visualizes the most important features 
used by the Random Forest model for intrusion detection.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import os

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_model_and_data():
    """Load the trained model and preprocessed data."""
    print("Loading model and data...")
    
    # Load the trained model
    model = joblib.load('intrusion_detection_model_unsw.pkl')
    
    # Load the preprocessed training data to get feature names
    train_data = pd.read_csv('train_processed.csv')
    feature_names = train_data.drop('label', axis=1).columns.tolist()
    
    print(f"✓ Model loaded successfully")
    print(f"✓ Total number of features: {len(feature_names)}\n")
    
    return model, train_data, feature_names

def extract_feature_importance(model, feature_names):
    """Extract and rank feature importances."""
    print("Extracting feature importances...")
    
    # Extract feature importances from the Random Forest model
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("✓ Feature importances extracted\n")
    
    # Display top 20 most important features
    print("=" * 60)
    print("Top 20 Most Important Features:")
    print("=" * 60)
    print(feature_importance_df.head(20).to_string(index=False))
    print("\n" + "=" * 60)
    print(f"Top 10 features account for {feature_importance_df.head(10)['Importance'].sum():.2%} of total importance")
    print(f"Top 20 features account for {feature_importance_df.head(20)['Importance'].sum():.2%} of total importance")
    print("=" * 60 + "\n")
    
    return feature_importance_df, importances

def plot_top_features_horizontal(feature_importance_df):
    """Create horizontal bar chart of top 20 features."""
    print("Creating horizontal bar chart...")
    
    plt.figure(figsize=(12, 8))
    top_20 = feature_importance_df.head(20)
    plt.barh(range(len(top_20)), top_20['Importance'], color='steelblue')
    plt.yticks(range(len(top_20)), top_20['Feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Top 20 Most Important Features for Intrusion Detection', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_top20.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: feature_importance_top20.png\n")

def plot_top_features_vertical(feature_importance_df):
    """Create vertical bar chart of top 15 features with values."""
    print("Creating vertical bar chart...")
    
    plt.figure(figsize=(14, 6))
    top_15 = feature_importance_df.head(15)
    bars = plt.bar(range(len(top_15)), top_15['Importance'], 
                   color='coral', edgecolor='black', linewidth=1.2)
    plt.xticks(range(len(top_15)), top_15['Feature'], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12, fontweight='bold')
    plt.ylabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('Top 15 Most Important Features with Importance Scores', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance_top15_vertical.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: feature_importance_top15_vertical.png\n")

def plot_cumulative_importance(feature_importance_df):
    """Create cumulative importance plot."""
    print("Creating cumulative importance plot...")
    
    plt.figure(figsize=(12, 6))
    cumulative_importance = np.cumsum(feature_importance_df['Importance'])
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
             linewidth=2.5, color='darkgreen', marker='o', markersize=3, markevery=5)
    plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
    plt.axhline(y=0.90, color='orange', linestyle='--', linewidth=2, label='90% Threshold')
    plt.xlabel('Number of Features', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
    plt.title('Cumulative Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Find how many features needed for 90% and 95% importance
    features_90 = np.argmax(cumulative_importance >= 0.90) + 1
    features_95 = np.argmax(cumulative_importance >= 0.95) + 1
    plt.axvline(x=features_90, color='orange', linestyle=':', alpha=0.5)
    plt.axvline(x=features_95, color='red', linestyle=':', alpha=0.5)
    
    plt.text(features_90, 0.5, f'{features_90} features\nfor 90%', 
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.text(features_95, 0.6, f'{features_95} features\nfor 95%', 
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('cumulative_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: cumulative_feature_importance.png")
    print(f"\nInsight: Only {features_90} features are needed to capture 90% of the model's predictive power")
    print(f"Insight: Only {features_95} features are needed to capture 95% of the model's predictive power\n")

def plot_correlation_heatmap(train_data, feature_importance_df):
    """Create correlation heatmap of top 10 important features."""
    print("Creating correlation heatmap...")
    
    top_10_features = feature_importance_df.head(10)['Feature'].tolist()
    train_data_subset = train_data[top_10_features]
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = train_data_subset.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap of Top 10 Most Important Features', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('top10_features_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: top10_features_correlation.png\n")

def print_summary_statistics(feature_importance_df, importances, feature_names):
    """Print summary statistics of feature importance."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nTotal Features: {len(feature_names)}")
    print(f"\nMost Important Feature: {feature_importance_df.iloc[0]['Feature']}")
    print(f"Importance Score: {feature_importance_df.iloc[0]['Importance']:.4f}")
    print(f"\nLeast Important Feature: {feature_importance_df.iloc[-1]['Feature']}")
    print(f"Importance Score: {feature_importance_df.iloc[-1]['Importance']:.6f}")
    print(f"\nMean Importance: {importances.mean():.4f}")
    print(f"Median Importance: {np.median(importances):.4f}")
    print(f"Std Deviation: {importances.std():.4f}")
    print(f"\nFeatures above mean importance: {sum(importances > importances.mean())}")
    print(f"Features below mean importance: {sum(importances < importances.mean())}")
    print("="*60 + "\n")

def save_results(feature_importance_df):
    """Save feature importance data to CSV."""
    print("Saving results...")
    feature_importance_df.to_csv('feature_importance_full.csv', index=False)
    print("✓ Saved: feature_importance_full.csv\n")

def main():
    """Main function to run the complete feature importance analysis."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS FOR INTRUSION DETECTION SYSTEM")
    print("="*60 + "\n")
    
    # Load model and data
    model, train_data, feature_names = load_model_and_data()
    
    # Extract feature importance
    feature_importance_df, importances = extract_feature_importance(model, feature_names)
    
    # Create visualizations
    print("Generating visualizations...\n")
    plot_top_features_horizontal(feature_importance_df)
    plot_top_features_vertical(feature_importance_df)
    plot_cumulative_importance(feature_importance_df)
    plot_correlation_heatmap(train_data, feature_importance_df)
    
    # Print summary statistics
    print_summary_statistics(feature_importance_df, importances, feature_names)
    
    # Save results
    save_results(feature_importance_df)
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - feature_importance_top20.png")
    print("  - feature_importance_top15_vertical.png")
    print("  - cumulative_feature_importance.png")
    print("  - top10_features_correlation.png")
    print("  - feature_importance_full.csv")
    print("\n")

if __name__ == "__main__":
    main()

