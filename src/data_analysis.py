import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='data/creditcard.csv'):
    """Load the credit card dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please ensure the dataset is in the data/ folder.")
        return None

def basic_statistics(df):
    """Display basic statistics about the dataset"""
    print("\n=== BASIC STATISTICS ===")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['Class'].sum()}")
    print(f"Legitimate transactions: {(df['Class'] == 0).sum()}")
    print(f"Fraud percentage: {(df['Class'].sum() / len(df) * 100):.2f}%")
    
    print("\n=== DATASET INFO ===")
    print(df.info())
    
    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

def visualize_class_distribution(df):
    """Visualize the distribution of fraudulent vs legitimate transactions"""
    plt.figure(figsize=(10, 6))
    
    # Class distribution
    plt.subplot(1, 2, 1)
    class_counts = df['Class'].value_counts()
    plt.pie(class_counts.values, labels=['Legitimate', 'Fraudulent'], autopct='%1.1f%%')
    plt.title('Transaction Class Distribution')
    
    # Amount distribution by class
    plt.subplot(1, 2, 2)
    plt.hist(df[df['Class'] == 0]['Amount'], alpha=0.7, label='Legitimate', bins=50)
    plt.hist(df[df['Class'] == 1]['Amount'], alpha=0.7, label='Fraudulent', bins=50)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    plt.title('Transaction Amount Distribution by Class')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('../report/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_features(df):
    """Analyze feature distributions and correlations"""
    # Select only numerical features (excluding 'Class' and 'Time')
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Class')
    numerical_features.remove('Time')
    
    print(f"\n=== FEATURE ANALYSIS ===")
    print(f"Number of numerical features: {len(numerical_features)}")
    
    # Feature statistics
    feature_stats = df[numerical_features].describe()
    print("\nFeature Statistics:")
    print(feature_stats)
    
    # Correlation with target variable
    correlations = df[numerical_features + ['Class']].corr()['Class'].sort_values(ascending=False)
    print("\nTop 10 features correlated with fraud:")
    print(correlations.head(10))
    
    # Visualize correlations
    plt.figure(figsize=(12, 8))
    top_features = correlations.head(10).index.tolist()
    correlation_matrix = df[top_features + ['Class']].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Top Features')
    plt.tight_layout()
    plt.savefig('../report/feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_time_patterns(df):
    """Analyze time-based patterns in fraud"""
    plt.figure(figsize=(15, 5))
    
    # Convert time to hours
    df['Hour'] = df['Time'] / 3600
    
    # Fraud by hour
    plt.subplot(1, 3, 1)
    fraud_by_hour = df[df['Class'] == 1]['Hour'].value_counts().sort_index()
    plt.plot(fraud_by_hour.index, fraud_by_hour.values, 'r-', linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Fraudulent Transactions')
    plt.title('Fraudulent Transactions by Hour')
    plt.grid(True, alpha=0.3)
    
    # Amount by hour
    plt.subplot(1, 3, 2)
    hourly_amounts = df.groupby('Hour')['Amount'].mean()
    plt.plot(hourly_amounts.index, hourly_amounts.values, 'b-', linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Transaction Amount')
    plt.title('Average Transaction Amount by Hour')
    plt.grid(True, alpha=0.3)
    
    # Fraud rate by hour
    plt.subplot(1, 3, 3)
    fraud_rate_by_hour = df.groupby('Hour')['Class'].mean()
    plt.plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values * 100, 'g-', linewidth=2)
    plt.xlabel('Hour of Day')
    plt.ylabel('Fraud Rate (%)')
    plt.title('Fraud Rate by Hour')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../report/time_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the analysis"""
    print("Credit Card Fraud Detection - Data Analysis")
    print("=" * 50)
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Perform analysis
    basic_statistics(df)
    visualize_class_distribution(df)
    analyze_features(df)
    analyze_time_patterns(df)
    
    print("\nAnalysis complete! Check the report/ folder for saved visualizations.")

if __name__ == "__main__":
    main()
