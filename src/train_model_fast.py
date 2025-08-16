import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='data/creditcard.csv'):
    """Load and prepare the dataset for training"""
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(['Class', 'Time'], axis=1)
    y = df['Class']
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_random_forest_fast(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier with default parameters for speed"""
    print("\n=== Training Random Forest (Fast Mode) ===")
    
    # Use default parameters for faster training
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1  # Use all CPU cores
    )
    
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    
    return rf

def train_logistic_regression_fast(X_train, y_train, X_test, y_test):
    """Train Logistic Regression classifier with default parameters"""
    print("\n=== Training Logistic Regression (Fast Mode) ===")
    
    # Use default parameters for faster training
    lr = LogisticRegression(
        random_state=42, 
        class_weight='balanced', 
        max_iter=1000
    )
    
    print("Training Logistic Regression...")
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    return lr

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Evaluate the model performance"""
    print(f"\n--- {model_name} Results ---")
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # ROC AUC score
    auc_score = roc_auc_score(y_true, y_pred_proba)
    print(f"ROC AUC Score: {auc_score:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraudulent'],
                yticklabels=['Legitimate', 'Fraudulent'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'report/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'report/{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, scaler, model_name, file_path='models/fraud_model.pkl'):
    """Save the trained model and scaler"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'model_name': model_name
    }
    
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {file_path}")

def main():
    """Main function to train the fraud detection model (Fast Mode)"""
    print("Credit Card Fraud Detection - Model Training (Fast Mode)")
    print("=" * 60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    # Train different models
    models = {}
    
    # Random Forest
    try:
        print("\n🚀 Starting Random Forest training...")
        rf_model = train_random_forest_fast(X_train, y_train, X_test, y_test)
        models['Random Forest'] = rf_model
        print("✅ Random Forest training completed!")
    except Exception as e:
        print(f"❌ Error training Random Forest: {e}")
    
    # Logistic Regression
    try:
        print("\n🚀 Starting Logistic Regression training...")
        lr_model = train_logistic_regression_fast(X_train, y_train, X_test, y_test)
        models['Logistic Regression'] = lr_model
        print("✅ Logistic Regression training completed!")
    except Exception as e:
        print(f"❌ Error training Logistic Regression: {e}")
    
    # Select best model based on ROC AUC
    best_model_name = None
    best_auc = 0
    
    for name, model in models.items():
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            if auc_score > best_auc:
                best_auc = auc_score
                best_model_name = name
        except:
            continue
    
    if best_model_name:
        print(f"\n🎯 BEST MODEL SELECTED ===")
        print(f"Model: {best_model_name}")
        print(f"ROC AUC: {best_auc:.4f}")
        
        # Save the best model
        save_model(models[best_model_name], scaler, best_model_name)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Your fraud detection model is now ready to use!")
        print(f"Run 'python src/predict.py' to test predictions.")
    else:
        print("\n❌ No models were successfully trained.")

if __name__ == "__main__":
    main()
