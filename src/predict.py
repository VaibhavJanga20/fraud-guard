import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path='models/fraud_model.pkl'):
    """Load the trained model and scaler"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        model_name = model_data['model_name']
        
        print(f"Model loaded successfully: {model_name}")
        return model, scaler, model_name
    
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return None, None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def prepare_sample_data():
    """Create sample transaction data for demonstration"""
    # Create sample features (28 V1-V28 features + Amount)
    np.random.seed(42)
    
    # Generate 10 sample transactions
    n_samples = 10
    n_features = 28
    
    # Generate V1-V28 features (normal distribution)
    features = np.random.randn(n_samples, n_features)
    
    # Generate Amount (positive values, skewed distribution)
    amounts = np.abs(np.random.exponential(100, n_samples))
    
    # Combine features and amount
    sample_data = np.column_stack([features, amounts])
    
    # Create column names
    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Create DataFrame
    sample_df = pd.DataFrame(sample_data, columns=feature_names)
    
    return sample_df

def predict_fraud(model, scaler, transaction_data):
    """Predict fraud for given transaction data"""
    try:
        # Scale the features
        scaled_data = scaler.transform(transaction_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        return prediction, prediction_proba
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def analyze_transaction(transaction_data, prediction, prediction_proba):
    """Analyze and display transaction details"""
    print("\n=== TRANSACTION ANALYSIS ===")
    
    for i in range(len(transaction_data)):
        print(f"\nTransaction {i+1}:")
        print(f"Amount: ${transaction_data.iloc[i]['Amount']:.2f}")
        
        # Show top 5 most important features (if Random Forest)
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': transaction_data.columns,
                'Value': transaction_data.iloc[i].values,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("Top 5 most important features:")
            for _, row in feature_importance.head().iterrows():
                print(f"  {row['Feature']}: {row['Value']:.4f} (importance: {row['Importance']:.4f})")
        
        # Prediction results
        fraud_prob = prediction_proba[i][1]
        legitimate_prob = prediction_proba[i][0]
        
        print(f"Fraud Probability: {fraud_prob:.4f} ({fraud_prob*100:.2f}%)")
        print(f"Legitimate Probability: {legitimate_prob:.4f} ({legitimate_prob*100:.2f}%)")
        
        if prediction[i] == 1:
            print("🚨 PREDICTION: FRAUDULENT TRANSACTION")
        else:
            print("✅ PREDICTION: LEGITIMATE TRANSACTION")
        
        # Risk assessment
        if fraud_prob > 0.8:
            risk_level = "HIGH RISK"
        elif fraud_prob > 0.5:
            risk_level = "MEDIUM RISK"
        elif fraud_prob > 0.2:
            risk_level = "LOW RISK"
        else:
            risk_level = "VERY LOW RISK"
        
        print(f"Risk Level: {risk_level}")

def interactive_prediction():
    """Interactive mode for user input"""
    print("\n=== INTERACTIVE PREDICTION MODE ===")
    print("Enter transaction details (or press Enter to use sample data):")
    
    try:
        # Get amount from user
        amount_input = input("Transaction Amount (USD): ").strip()
        
        if amount_input:
            amount = float(amount_input)
            
            # Generate random features for the amount
            np.random.seed()
            features = np.random.randn(1, 28)
            transaction_data = np.column_stack([features, [amount]])
            
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']
            transaction_df = pd.DataFrame(transaction_data, columns=feature_names)
            
            # Make prediction
            prediction, prediction_proba = predict_fraud(model, scaler, transaction_df)
            
            if prediction is not None:
                analyze_transaction(transaction_df, prediction, prediction_proba)
        else:
            print("Using sample data...")
            sample_data = prepare_sample_data()
            prediction, prediction_proba = predict_fraud(model, scaler, sample_data)
            
            if prediction is not None:
                analyze_transaction(sample_data, prediction, prediction_proba)
    
    except ValueError:
        print("Invalid amount. Please enter a valid number.")
    except KeyboardInterrupt:
        print("\nExiting interactive mode.")

def batch_prediction(file_path):
    """Make predictions on a batch of transactions from CSV file"""
    try:
        print(f"\n=== BATCH PREDICTION FROM {file_path} ===")
        
        # Load transaction data
        transactions = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        missing_columns = [col for col in required_columns if col not in transactions.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            print("Please ensure your CSV file contains all required features.")
            return
        
        # Make predictions
        prediction, prediction_proba = predict_fraud(model, scaler, transactions)
        
        if prediction is not None:
            # Add predictions to the dataframe
            transactions['Fraud_Prediction'] = prediction
            transactions['Fraud_Probability'] = prediction_proba[:, 1]
            transactions['Legitimate_Probability'] = prediction_proba[:, 0]
            
            # Save results
            output_file = file_path.replace('.csv', '_with_predictions.csv')
            transactions.to_csv(output_file, index=False)
            print(f"Predictions saved to: {output_file}")
            
            # Summary statistics
            fraud_count = prediction.sum()
            total_count = len(prediction)
            print(f"\nBatch Prediction Summary:")
            print(f"Total transactions: {total_count}")
            print(f"Predicted fraud: {fraud_count}")
            print(f"Fraud rate: {fraud_count/total_count*100:.2f}%")
            
            # Show high-risk transactions
            high_risk = transactions[transactions['Fraud_Probability'] > 0.8]
            if len(high_risk) > 0:
                print(f"\nHigh-risk transactions (fraud probability > 80%): {len(high_risk)}")
                print(high_risk[['Amount', 'Fraud_Probability']].head())
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"Error processing batch file: {e}")

def main():
    """Main function for fraud prediction"""
    print("Credit Card Fraud Detection - Prediction System")
    print("=" * 50)
    
    # Load the trained model
    global model, scaler, model_name
    model, scaler, model_name = load_model()
    
    if model is None:
        return
    
    while True:
        print("\n" + "="*50)
        print("PREDICTION MENU:")
        print("1. Use sample data for demonstration")
        print("2. Interactive prediction (enter amount)")
        print("3. Batch prediction from CSV file")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Use sample data
            sample_data = prepare_sample_data()
            prediction, prediction_proba = predict_fraud(model, scaler, sample_data)
            
            if prediction is not None:
                analyze_transaction(sample_data, prediction, prediction_proba)
        
        elif choice == '2':
            # Interactive mode
            interactive_prediction()
        
        elif choice == '3':
            # Batch prediction
            file_path = input("Enter CSV file path: ").strip()
            if file_path:
                batch_prediction(file_path)
            else:
                print("No file path provided.")
        
        elif choice == '4':
            print("Exiting prediction system. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
