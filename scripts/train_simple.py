#!/usr/bin/env python3
"""
Simple training script that works with all the models.
This script bypasses complex import issues by using direct sklearn implementations.
"""

import sys
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MLflow imports
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_mlflow():
    """Setup MLflow tracking."""
    mlflow.set_tracking_uri("./mlruns")
    
    # Create or get experiment
    experiment_name = "california_housing_prediction"
    try:
        experiment = mlflow.create_experiment(experiment_name)
    except:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    return experiment

def load_data():
    """Load the California Housing dataset."""
    print("🔄 Loading California Housing dataset...")
    
    # Try to load from processed data first
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        test_df = pd.read_csv("data/processed/test.csv")
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col != 'target']
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        
        print(f"✅ Loaded processed data: Train {X_train.shape}, Test {X_test.shape}")
        
    except:
        # Fallback to sklearn dataset
        from sklearn.datasets import fetch_california_housing
        
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        # Create feature names
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name='target')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"✅ Loaded sklearn data: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression model."""
    print("\n🚀 Training Linear Regression...")
    
    with mlflow.start_run(run_name="linear_regression") as run:
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("algorithm", "Linear Regression")
        mlflow.log_param("scaler", "StandardScaler")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="linear_regression"
        )
        
        # Save scaler
        joblib.dump(scaler, "models/linear_regression_scaler.pkl")
        mlflow.log_artifact("models/linear_regression_scaler.pkl")
        
        # Add tags
        mlflow.set_tag("model_type", "linear_regression")
        mlflow.set_tag("algorithm", "Linear Regression")
        
        print(f"✅ Linear Regression - R²: {r2:.3f}, RMSE: {rmse:.3f}")
        
        return {"model": model, "scaler": scaler, "metrics": {"r2": r2, "rmse": rmse, "mae": mae}}

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model."""
    print("\n🚀 Training Random Forest...")
    
    with mlflow.start_run(run_name="random_forest") as run:
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("algorithm", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="random_forest"
        )
        
        # Add tags
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("algorithm", "Random Forest")
        
        print(f"✅ Random Forest - R²: {r2:.3f}, RMSE: {rmse:.3f}")
        
        return {"model": model, "metrics": {"r2": r2, "rmse": rmse, "mae": mae}}

def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost model."""
    print("\n🚀 Training XGBoost...")
    
    try:
        import xgboost as xgb
        
        with mlflow.start_run(run_name="xgboost") as run:
            # Train model
            try:
                # Try GPU first
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='hist',
                    device='cuda'
                )
            except:
                # Fallback to CPU
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    tree_method='hist'
                )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "xgboost")
            mlflow.log_param("algorithm", "XGBoost")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.1)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Log model
            mlflow.xgboost.log_model(
                model, 
                "model",
                registered_model_name="xgboost"
            )
            
            # Add tags
            mlflow.set_tag("model_type", "xgboost")
            mlflow.set_tag("algorithm", "XGBoost")
            
            print(f"✅ XGBoost - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            
            return {"model": model, "metrics": {"r2": r2, "rmse": rmse, "mae": mae}}
            
    except ImportError:
        print("⚠️  XGBoost not available, skipping...")
        return None

def train_lightgbm(X_train, X_test, y_train, y_test):
    """Train LightGBM model."""
    print("\n🚀 Training LightGBM...")
    
    try:
        import lightgbm as lgb
        
        with mlflow.start_run(run_name="lightgbm") as run:
            # Train model
            try:
                # Try GPU first
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    device='gpu'
                )
            except:
                # Fallback to CPU
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    device='cpu'
                )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "lightgbm")
            mlflow.log_param("algorithm", "LightGBM")
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.1)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Log model
            mlflow.lightgbm.log_model(
                model, 
                "model",
                registered_model_name="lightgbm"
            )
            
            # Add tags
            mlflow.set_tag("model_type", "lightgbm")
            mlflow.set_tag("algorithm", "LightGBM")
            
            print(f"✅ LightGBM - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            
            return {"model": model, "metrics": {"r2": r2, "rmse": rmse, "mae": mae}}
            
    except ImportError:
        print("⚠️  LightGBM not available, skipping...")
        return None

def train_neural_network(X_train, X_test, y_train, y_test):
    """Train a simple neural network."""
    print("\n🚀 Training Neural Network...")
    
    try:
        from sklearn.neural_network import MLPRegressor
        
        with mlflow.start_run(run_name="neural_network") as run:
            # Scale the features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "neural_network")
            mlflow.log_param("algorithm", "Neural Network")
            mlflow.log_param("hidden_layers", "(100, 50)")
            mlflow.log_param("max_iter", 500)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="neural_network"
            )
            
            # Save scaler
            joblib.dump(scaler, "models/neural_network_scaler.pkl")
            mlflow.log_artifact("models/neural_network_scaler.pkl")
            
            # Add tags
            mlflow.set_tag("model_type", "neural_network")
            mlflow.set_tag("algorithm", "Neural Network")
            
            print(f"✅ Neural Network - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            
            return {"model": model, "scaler": scaler, "metrics": {"r2": r2, "rmse": rmse, "mae": mae}}
            
    except Exception as e:
        print(f"⚠️  Neural Network training failed: {e}")
        return None

def main():
    """Main training function."""
    print("🎯 Starting California Housing Model Training")
    print("=" * 70)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train all models
    results = {}
    
    start_time = time.time()
    
    # Train each model
    results['linear_regression'] = train_linear_regression(X_train, X_test, y_train, y_test)
    results['random_forest'] = train_random_forest(X_train, X_test, y_train, y_test)
    results['xgboost'] = train_xgboost(X_train, X_test, y_train, y_test)
    results['lightgbm'] = train_lightgbm(X_train, X_test, y_train, y_test)
    results['neural_network'] = train_neural_network(X_train, X_test, y_train, y_test)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    successful_models = []
    for model_name, result in results.items():
        if result:
            metrics = result['metrics']
            print(f"🏆 {model_name.upper():<15} - R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.3f}")
            successful_models.append(model_name)
        else:
            print(f"❌ {model_name.upper():<15} - Failed to train")
    
    print("=" * 70)
    print(f"✅ {len(successful_models)}/{len(results)} models trained successfully")
    print(f"⏱️  Total training time: {total_time:.1f} seconds")
    print(f"🗂️  Models registered in MLflow: {len(successful_models)}")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main()) 