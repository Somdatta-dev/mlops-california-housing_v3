#!/usr/bin/env python3
"""
Database initialization script for MLOps pipeline.

This script initializes the database, creates tables, and sets up initial data.
"""

import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database import init_database, get_database_manager
from database.models import ModelVersion
from utils.logging_config import setup_logging, get_logger
from api.config import get_config

def setup_initial_data():
    """Set up initial data in the database."""
    logger = get_logger(__name__)
    
    try:
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Check if we already have model versions
            existing_models = session.query(ModelVersion).count()
            
            if existing_models == 0:
                logger.info("Setting up initial model data...")
                
                # Add our trained models to the database
                models_to_add = [
                    {
                        "model_name": "linear_regression",
                        "version": "1",
                        "algorithm": "Linear Regression",
                        "feature_names": [
                            "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                            "Population", "AveOccup", "Latitude", "Longitude"
                        ],
                        "training_r2": 0.5758,
                        "training_rmse": 0.7197,
                        "training_mae": 0.5286,
                        "test_r2": 0.5758,
                        "test_rmse": 0.7456,
                        "test_mae": 0.5332,
                        "hyperparameters": {
                            "fit_intercept": True,
                            "normalize": False
                        },
                        "gpu_accelerated": False
                    },
                    {
                        "model_name": "random_forest",
                        "version": "1", 
                        "algorithm": "Random Forest",
                        "feature_names": [
                            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                            "Population", "AveOccup", "Latitude", "Longitude"
                        ],
                        "training_r2": 0.8221,
                        "training_rmse": 0.4871,
                        "training_mae": 0.3110,
                        "test_r2": 0.8037,
                        "test_rmse": 0.5072,
                        "test_mae": 0.3303,
                        "hyperparameters": {
                            "n_estimators": 50,
                            "random_state": 42,
                            "max_depth": None,
                            "min_samples_split": 2,
                            "min_samples_leaf": 1
                        },
                        "gpu_accelerated": False
                    }
                ]
                
                for model_data in models_to_add:
                    model_version = ModelVersion(**model_data)
                    session.add(model_version)
                
                session.commit()
                logger.info(f"Added {len(models_to_add)} initial model versions")
                
            else:
                logger.info(f"Database already contains {existing_models} model versions")
                
    except Exception as e:
        logger.error(f"Failed to set up initial data: {e}")
        raise

def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(description="Initialize MLOps database")
    parser.add_argument("--database-url", help="Database URL override")
    parser.add_argument("--echo", action="store_true", help="Echo SQL statements")
    parser.add_argument("--reset", action="store_true", help="Reset database (drop and recreate tables)")
    parser.add_argument("--initial-data", action="store_true", help="Set up initial data")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    try:
        # Get configuration
        config = get_config()
        database_url = args.database_url or config.database_url
        echo = args.echo or config.database_echo
        
        logger.info("🚀 Initializing MLOps database...")
        logger.info(f"Database URL: {database_url}")
        logger.info(f"Echo SQL: {echo}")
        
        # Initialize database
        db_manager = init_database(database_url, echo)
        
        if args.reset:
            logger.warning("⚠️  Resetting database (dropping all tables)...")
            db_manager.drop_tables()
            db_manager.create_tables()
            logger.info("✅ Database reset completed")
        
        # Test connection
        if db_manager.health_check():
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection failed")
            return 1
        
        # Get connection info
        conn_info = db_manager.get_connection_info()
        logger.info(f"Database info: {conn_info}")
        
        # Set up initial data if requested
        if args.initial_data:
            setup_initial_data()
        
        logger.info("🎉 Database initialization completed successfully!")
        
        # Show table counts
        with db_manager.get_session() as session:
            from database.models import ModelVersion, PredictionLog, PerformanceMetrics, SystemHealth
            
            model_count = session.query(ModelVersion).count()
            prediction_count = session.query(PredictionLog).count()
            metrics_count = session.query(PerformanceMetrics).count()
            health_count = session.query(SystemHealth).count()
            
            logger.info(f"📊 Table counts:")
            logger.info(f"   Model Versions: {model_count}")
            logger.info(f"   Prediction Logs: {prediction_count}")
            logger.info(f"   Performance Metrics: {metrics_count}")
            logger.info(f"   System Health: {health_count}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 