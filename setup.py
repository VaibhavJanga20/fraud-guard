#!/usr/bin/env python3
"""
Credit Card Fraud Detection Project Setup Script
This script helps set up the project environment and install dependencies.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version)
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'report']
    
    print("\n📁 Creating project directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"ℹ️  Directory already exists: {directory}")

def check_dataset():
    """Check if the dataset is present"""
    dataset_path = os.path.join('data', 'creditcard.csv')
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found: {dataset_path}")
        return True
    else:
        print(f"⚠️  Dataset not found: {dataset_path}")
        print("📥 Please download the dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("   and place it in the 'data/' folder.")
        return False

def run_tests():
    """Run basic tests to ensure everything is working"""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        import matplotlib
        import seaborn
        print("✅ All packages imported successfully!")
        
        # Test data analysis script
        if os.path.exists('src/data_analysis.py'):
            print("✅ Data analysis script found!")
        else:
            print("❌ Data analysis script not found!")
            
        # Test training script
        if os.path.exists('src/train_model.py'):
            print("✅ Training script found!")
        else:
            print("❌ Training script not found!")
            
        # Test prediction script
        if os.path.exists('src/predict.py'):
            print("✅ Prediction script found!")
        else:
            print("❌ Prediction script not found!")
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Credit Card Fraud Detection Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Some packages may not have been installed correctly.")
    
    # Check dataset
    dataset_available = check_dataset()
    
    # Run tests
    tests_passed = run_tests()
    
    print("\n" + "=" * 50)
    print("🎯 Setup Summary:")
    print(f"✅ Python version: Compatible")
    print(f"✅ Directories: Created")
    print(f"✅ Packages: Installed")
    print(f"{'✅' if dataset_available else '⚠️'} Dataset: {'Available' if dataset_available else 'Not found'}")
    print(f"{'✅' if tests_passed else '❌'} Tests: {'Passed' if tests_passed else 'Failed'}")
    
    if dataset_available and tests_passed:
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run data analysis: cd src && python data_analysis.py")
        print("2. Train the model: python train_model.py")
        print("3. Make predictions: python predict.py")
    else:
        print("\n⚠️  Setup completed with warnings.")
        if not dataset_available:
            print("   - Download the dataset before proceeding")
        if not tests_passed:
            print("   - Check package installation and try again")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
