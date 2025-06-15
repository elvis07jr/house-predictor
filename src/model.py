import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def split_data(df, test_size=0.2):
    """Step 6: Split data"""
    X = df.drop('target', axis=1)
    y = df['target']
    return train_test_split(X, y, test_size=test_size, random_state=42)

def create_model():
    """Step 7: Create ML pipeline - matches your thinking!"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

def train_model(X_train, y_train):
    """Step 8: Train model"""
    model = create_model()
    print("Training model...")
    model.fit(X_train, y_train)
    print("✓ Training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """Step 9: Evaluate model"""
    predictions = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, predictions),
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'r2': r2_score(y_test, predictions)
    }
    
    print(f"✓ Model Performance:")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  R²: {metrics['r2']:.3f}")
    
    return metrics, predictions

def save_model(model, filepath='model.pkl'):
    """Step 10: Save model"""
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")
    return filepath

def ml_pipeline():
    """Complete ML training pipeline - your step-by-step style!"""
    from src.data_processing import data_pipeline
    from src.feature_engineering import feature_pipeline
    
    print("Starting ML Pipeline...")
    
    # Step-by-step pipeline
    print("\n1. Loading and cleaning data...")
    raw_data = data_pipeline()
    
    print("\n2. Engineering features...")
    processed_data = feature_pipeline(raw_data)
    
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = split_data(processed_data)
    
    print("\n4. Training model...")
    model = train_model(X_train, y_train)
    
    print("\n5. Evaluating model...")
    metrics, predictions = evaluate_model(model, X_test, y_test)
    
    print("\n6. Saving model...")
    save_model(model)
    
    print("\n🎉 Pipeline complete!")
    return model, metrics