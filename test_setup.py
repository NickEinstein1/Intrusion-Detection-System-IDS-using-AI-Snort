"""
Test script to verify the setup for feature importance analysis.
This checks if all required files and packages are available.
"""

import sys
import os

def check_packages():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'joblib',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages are installed!")
        return True

def check_files():
    """Check if required files exist."""
    print("\nChecking required files...")
    required_files = {
        'intrusion_detection_model_unsw.pkl': 'Trained model file',
        'train_processed.csv': 'Preprocessed training data (run notebook first)',
        'Intrusion_Detection_System(IDS).ipynb': 'Main Jupyter notebook'
    }
    
    missing_files = []
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {filename} - NOT FOUND ({description})")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        if 'train_processed.csv' in missing_files:
            print("\n📝 Note: You need to run the Jupyter notebook first to generate the preprocessed data.")
            print("   Open 'Intrusion_Detection_System(IDS).ipynb' and run all cells up to 'Model Training'.")
        return False
    else:
        print("\n✓ All required files are present!")
        return True

def main():
    """Run all checks."""
    print("="*60)
    print("SETUP VERIFICATION FOR FEATURE IMPORTANCE ANALYSIS")
    print("="*60 + "\n")
    
    packages_ok = check_packages()
    files_ok = check_files()
    
    print("\n" + "="*60)
    if packages_ok and files_ok:
        print("✅ SETUP COMPLETE - Ready to run feature importance analysis!")
        print("="*60)
        print("\nYou can now run:")
        print("  python feature_importance_analysis.py")
        print("\nOr open the Jupyter notebook and run the Feature Importance section.")
        return 0
    else:
        print("❌ SETUP INCOMPLETE - Please fix the issues above")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())

