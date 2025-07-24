#!/usr/bin/env python3
"""
Script to setup and test the data management system.
This script downloads, validates, and preprocesses the California Housing dataset.
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_manager import DataManager
from utils.logging_config import setup_logging


def main():
    """Main function to setup data management system."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting data setup process")
    
    try:
        # Initialize DataManager
        data_manager = DataManager(data_dir="data", use_dvc=False)  # Disable DVC for now
        
        # Load and validate data
        logger.info("Loading California Housing dataset...")
        df = data_manager.load_data()
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Validate data quality
        logger.info("Validating data quality...")
        quality_report = data_manager.validate_data(df)
        logger.info(f"Data quality score: {quality_report.quality_score:.2f}%")
        logger.info(f"Valid records: {quality_report.valid_records}/{quality_report.total_records}")
        
        if quality_report.invalid_records > 0:
            logger.warning(f"Found {quality_report.invalid_records} invalid records")
            for error in quality_report.validation_errors[:5]:  # Show first 5 errors
                logger.warning(f"Record {error['record_id']}: {error['error']}")
        
        # Save data profile
        data_manager.save_data_profile(quality_report.data_profile)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        train_df, test_df = data_manager.preprocess_data(df)
        logger.info(f"Train set: {train_df.shape}, Test set: {test_df.shape}")
        
        # Test data loading
        logger.info("Testing processed data loading...")
        train_loaded, test_loaded = data_manager.load_processed_data()
        assert train_loaded.shape == train_df.shape
        assert test_loaded.shape == test_df.shape
        logger.info("Data loading test passed")
        
        # Test inference data preparation
        logger.info("Testing inference data preparation...")
        sample_data = {
            'MedInc': 8.3252,
            'HouseAge': 41.0,
            'AveRooms': 6.984,
            'AveBedrms': 1.024,
            'Population': 322.0,
            'AveOccup': 2.555,
            'Latitude': 37.88,
            'Longitude': -122.23
        }
        
        inference_array = data_manager.prepare_inference_data(sample_data)
        logger.info(f"Inference array shape: {inference_array.shape}")
        
        # Show data summary
        summary = data_manager.get_data_summary()
        logger.info("Data Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("Data setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data setup: {e}")
        raise


if __name__ == "__main__":
    main() 