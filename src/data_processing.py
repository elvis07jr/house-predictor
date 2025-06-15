import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

def load_data():
    """Step 1: Load California housing data"""
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    return df

def clean_data(df):
    """Step 2: Clean data - your pipeline style"""
    return (df
        .dropna()
        .query('HouseAge >= 0')  # Remove invalid house ages
        .query('target > 0')     # Remove invalid prices
        .query('Population > 0') # Remove invalid population
        .reset_index(drop=True)
    )

def validate_data(df):
    """Step 3: Data validation"""
    assert not df.isnull().any().any(), "Found null values"
    assert len(df) > 1000, "Dataset too small"
    print(f"✓ Data validated: {len(df)} rows, {len(df.columns)} columns")
    return df

def data_pipeline():
    """Complete data processing pipeline"""
    return (load_data()
        .pipe(clean_data)
        .pipe(validate_data)
    )