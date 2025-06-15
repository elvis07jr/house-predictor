import pandas as pd
import numpy as np

def create_features(df, mode='predict'):
    """Step 4: Feature engineering - your modular approach
    
    Args:
        df: Input dataframe
        mode: 'train' or 'predict' - determines which features to create
    """
    # Base features that don't depend on target - SAME FOR BOTH MODES
    df_features = df.assign(
        # Create meaningful features
        rooms_per_household=lambda x: x.AveRooms,
        bedrooms_per_room=lambda x: x.AveBedrms / x.AveRooms,
        population_per_room=lambda x: x.Population / x.AveRooms,
        # Location features
        lat_long_interaction=lambda x: x.Latitude * x.Longitude,
        # Income per room (economic density)
        income_per_room=lambda x: x.MedInc / x.AveRooms,
        # Population per household (more useful than population per room)
        population_per_household=lambda x: x.Population / x.AveOccup
    )
    
    # DON'T create target-dependent features - they cause prediction issues
    # Removed: price_per_room feature to ensure train/predict consistency
    
    return df_features

def select_features(df, mode='predict'):
    """Step 5: Feature selection"""
    # Feature columns that work for BOTH training and prediction
    feature_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude',
        'rooms_per_household', 'bedrooms_per_room', 
        'population_per_room', 'lat_long_interaction', 
        'income_per_room', 'population_per_household'
    ]
    
    if mode == 'train' and 'target' in df.columns:
        # For training, include target column
        return df[feature_cols + ['target']]
    else:
        # For prediction, only return feature columns
        return df[feature_cols]

def feature_pipeline(df, mode='predict'):
    """Complete feature engineering pipeline
    
    Args:
        df: Input dataframe
        mode: 'train' or 'predict'
    """
    return (df
        .pipe(create_features, mode=mode)
        .pipe(select_features, mode=mode)
    )