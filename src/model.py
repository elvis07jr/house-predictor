import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from src.data_processing import data_pipeline
from src.feature_engineering import create_features, select_features

def split_data(df):
    """Split features and target"""
    # Ensure target column exists
    if 'target' not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise ValueError("Target column 'target' not found in the data")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("4. Training Random Forest model...")
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("5. Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"✓ Model Performance:")
    print(f"  - RMSE: ${rmse:,.2f}")
    print(f"  - R² Score: {r2:.4f}")
    
    return metrics

def save_model(model, filename='model.pkl'):
    """Save the trained model"""
    print("6. Saving model...")
    joblib.dump(model, filename)
    print(f"✓ Model saved as {filename}")

def ml_pipeline():
    """Complete ML pipeline"""
    print("Starting ML Pipeline...")
    
    # Step 1: Load and validate data
    print("1. Loading and cleaning data...")
    raw_data = data_pipeline()
    print(f"✓ Data loaded and cleaned: {raw_data.shape[0]} rows, {raw_data.shape[1]} columns")
    
    # Step 2: Feature engineering - IMPORTANT: Use 'train' mode
    print("2. Engineering features...")
    features_df = create_features(raw_data, mode='train')  # Add mode='train'
    processed_data = select_features(features_df, mode='train')  # Add mode='train'
    
    print(f"✓ Features engineered: {processed_data.shape[1]-1} features")
    print("Available columns after feature engineering:", processed_data.columns.tolist())
    
    # Step 3: Split data
    print("3. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(processed_data)
    print(f"✓ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Step 6: Save model
    save_model(model)
    
    return model, metrics

if __name__ == "__main__":
    model, metrics = ml_pipeline()