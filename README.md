# Credit Card Fraud Detection Project

## Overview
This project implements a machine learning system for detecting fraudulent credit card transactions. The system uses various algorithms including Random Forest, Logistic Regression, and Support Vector Machine to classify transactions as legitimate or fraudulent.

## Project Structure
```
creditcard-fraud-detection/
│
├── data/
│   └── creditcard.csv          # Dataset (from Kaggle)
│
├── src/
│   ├── data_analysis.py        # Statistics + Visualizations
│   ├── train_model.py          # Training + Evaluation
│   └── predict.py              # Load Model & Predict
│
├── models/
│   └── fraud_model.pkl         # Saved Trained Model
│
├── report/
│   └── project_report.docx     # Course Report
│
└── README.md                   # This file
```

## Dataset
The project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle, which contains:
- 284,807 transactions
- 28 anonymized features (V1-V28)
- Transaction amount and timestamp
- Binary classification: 0 (legitimate) or 1 (fraudulent)

## Features
- **Data Analysis**: Comprehensive statistical analysis and visualizations
- **Model Training**: Multiple ML algorithms with hyperparameter tuning
- **Prediction System**: Interactive and batch prediction capabilities
- **Performance Evaluation**: ROC curves, confusion matrices, and detailed metrics

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Packages
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Alternative: Install from requirements.txt
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Analysis
Run the data analysis script to explore the dataset:
```bash
cd src
python data_analysis.py
```

This will:
- Load and analyze the dataset
- Generate visualizations (saved in `report/` folder)
- Display statistical summaries

### 2. Model Training
Train the fraud detection model:
```bash
cd src
python train_model.py
```

This will:
- Train multiple algorithms (Random Forest, Logistic Regression, SVM)
- Perform hyperparameter tuning
- Evaluate model performance
- Save the best model to `models/fraud_model.pkl`

### 3. Making Predictions
Use the trained model for predictions:
```bash
cd src
python predict.py
```

The prediction system offers:
- **Sample Data Demo**: Test with generated sample transactions
- **Interactive Mode**: Enter transaction amounts manually
- **Batch Processing**: Process multiple transactions from CSV files

## Model Performance

### Algorithms Tested
1. **Random Forest**: Ensemble method with feature importance
2. **Logistic Regression**: Linear classifier with regularization
3. **Support Vector Machine**: Non-linear classification

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

## Key Findings

### Data Characteristics
- **Class Imbalance**: Only ~0.17% of transactions are fraudulent
- **Feature Engineering**: 28 anonymized features derived from PCA
- **Amount Distribution**: Skewed distribution with most transactions being small amounts

### Model Insights
- **Feature Importance**: V17, V14, V12 are among the most important features
- **Performance**: Random Forest typically achieves the best ROC AUC scores
- **Scalability**: SVM may require data subsetting for large datasets

## File Descriptions

### `src/data_analysis.py`
- Dataset loading and validation
- Statistical analysis and summaries
- Visualization generation (class distribution, correlations, time patterns)
- Saves plots to `report/` folder

### `src/train_model.py`
- Data preprocessing and scaling
- Multiple model training with GridSearchCV
- Cross-validation and hyperparameter tuning
- Model evaluation and comparison
- Saves best model to `models/` folder

### `src/predict.py`
- Model loading and validation
- Interactive prediction interface
- Batch processing capabilities
- Risk assessment and detailed analysis

## Output Files

### Generated Visualizations
- `class_distribution.png`: Transaction class distribution
- `feature_correlations.png`: Feature correlation heatmap
- `time_patterns.png`: Time-based fraud patterns
- `*_confusion_matrix.png`: Model confusion matrices
- `*_roc_curve.png`: ROC curves for each model

### Model Files
- `fraud_model.pkl`: Serialized trained model with scaler

### Prediction Results
- `*_with_predictions.csv`: Input data with added prediction columns

## Troubleshooting

### Common Issues
1. **Dataset not found**: Ensure `creditcard.csv` is in the `data/` folder
2. **Model file missing**: Run `train_model.py` first to create the model
3. **Memory errors**: SVM training may require reducing dataset size
4. **Import errors**: Install required packages using pip

### Performance Tips
- Use smaller dataset subsets for faster experimentation
- Adjust hyperparameter grids for quicker training
- Consider using joblib for parallel processing

## Future Improvements

### Model Enhancements
- Implement deep learning approaches (Neural Networks)
- Add anomaly detection algorithms
- Feature engineering and selection techniques

### System Improvements
- Web interface for real-time predictions
- API endpoints for integration
- Real-time model updating capabilities

### Data Considerations
- Handle real-time streaming data
- Implement concept drift detection
- Add more transaction metadata

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License
This project is for educational purposes. Please ensure compliance with the original dataset license from Kaggle.

## Contact
For questions or issues, please open an issue in the repository.

## Acknowledgments
- Kaggle for providing the dataset
- Scikit-learn community for the ML library
- Open source contributors for supporting libraries

---

**Note**: This project is designed for educational and research purposes. In production environments, additional security measures, data validation, and model monitoring would be required.
