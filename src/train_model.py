import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest classifier"""
    print("\n=== Training Random Forest ===")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Grid search with cross-validation
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
    
    return best_rf

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression classifier"""
    print("\n=== Training Logistic Regression ===")
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    # Grid search
    lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    grid_search = GridSearchCV(lr, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_lr = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_lr.predict(X_test)
    y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_pred_proba, "Logistic Regression")
    
    return best_lr

def train_svm(X_train, y_train, X_test, y_test):
    """Train Support Vector Machine classifier"""
    print("\n=== Training SVM ===")
    
    # Use a smaller subset for SVM due to computational complexity
    if len(X_train) > 10000:
        indices = np.random.choice(len(X_train), 10000, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train.iloc[indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    }
    
    # Grid search
    svm = SVC(random_state=42, class_weight='balanced', probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_subset, y_train_subset)
    
    best_svm = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_svm.predict(X_test)
    y_pred_proba = best_svm.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    evaluate_model(y_test, y_pred, y_pred_proba, "SVM")
    
    return best_svm

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
    plt.savefig(f'../report/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
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
    plt.savefig(f'../report/{model_name.lower().replace(" ", "_")}_roc_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, scaler, model_name, file_path='../models/fraud_model.pkl'):
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
    """Main function to train the fraud detection model"""
    print("Credit Card Fraud Detection - Model Training")
    print("=" * 50)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
    
    # Train different models
    models = {}
    
    # Random Forest
    try:
        rf_model = train_random_forest(X_train, y_train, X_test, y_test)
        models['Random Forest'] = rf_model
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    # Logistic Regression
    try:
        lr_model = train_logistic_regression(X_train, y_train, X_test, y_test)
        models['Logistic Regression'] = lr_model
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
    
    # SVM
    try:
        svm_model = train_svm(X_train, y_train, X_test, y_test)
        models['SVM'] = svm_model
    except Exception as e:
        print(f"Error training SVM: {e}")
    
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
        print(f"\n=== BEST MODEL SELECTED ===")
        print(f"Model: {best_model_name}")
        print(f"ROC AUC: {best_auc:.4f}")
        
        # Save the best model
        save_model(models[best_model_name], scaler, best_model_name)
    else:
        print("\nNo models were successfully trained.")

if __name__ == "__main__":
    main()
