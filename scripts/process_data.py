#!/usr/bin/env python3
"""
Script to run the comprehensive data processing pipeline.
"""

import sys
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.data_pipeline import CaliforniaHousingPipeline, create_pipeline_config
from data.data_utils import create_data_summary_report
from utils.logging_config import setup_logging, get_logger


def main():
    """Main data processing script."""
    parser = argparse.ArgumentParser(description="Run California Housing data processing pipeline")
    
    # Data paths
    parser.add_argument("--raw-data", default="data/raw/california_housing.csv",
                       help="Path to raw data file")
    parser.add_argument("--processed-dir", default="data/processed",
                       help="Directory for processed data")
    parser.add_argument("--interim-dir", default="data/interim", 
                       help="Directory for interim data")
    
    # Processing options
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Test set size ratio (default: 0.2)")
    parser.add_argument("--validation-size", type=float, default=0.2,
                       help="Validation set size ratio (default: 0.2)")
    parser.add_argument("--scaler-type", default="standard",
                       choices=["standard", "robust", "minmax"],
                       help="Type of scaler to use")
    
    # Feature engineering
    parser.add_argument("--disable-feature-engineering", action="store_true",
                       help="Disable feature engineering")
    parser.add_argument("--disable-outlier-detection", action="store_true",
                       help="Disable outlier detection")
    parser.add_argument("--outlier-threshold", type=float, default=3.0,
                       help="Threshold for outlier detection")
    
    # Quality and validation
    parser.add_argument("--disable-validation", action="store_true",
                       help="Disable data quality validation")
    parser.add_argument("--missing-threshold", type=float, default=0.1,
                       help="Maximum allowed missing values ratio")
    
    # Output options
    parser.add_argument("--generate-report", action="store_true", default=True,
                       help="Generate comprehensive data report")
    parser.add_argument("--report-path", default="data/processed/data_summary_report.json",
                       help="Path for data summary report")
    
    # Logging
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--log-file", 
                       help="Log file path (default: logs/data_processing.log)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or "logs/data_processing.log"
    setup_logging(log_level=args.log_level, log_file=log_file)
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting California Housing data processing pipeline")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Create pipeline configuration
        config = create_pipeline_config(
            raw_data_path=args.raw_data,
            processed_data_path=args.processed_dir,
            interim_data_path=args.interim_dir,
            test_size=args.test_size,
            validation_size=args.validation_size,
            scaler_type=args.scaler_type,
            enable_feature_engineering=not args.disable_feature_engineering,
            enable_outlier_detection=not args.disable_outlier_detection,
            outlier_threshold=args.outlier_threshold,
            enable_data_validation=not args.disable_validation,
            missing_threshold=args.missing_threshold
        )
        
        logger.info("Pipeline configuration created successfully")
        
        # Initialize and run pipeline
        pipeline = CaliforniaHousingPipeline(config)
        
        start_time = time.time()
        train_df, val_df, test_df = pipeline.run_pipeline()
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Pipeline completed successfully in {processing_time:.2f} seconds")
        
        # Generate comprehensive report
        if args.generate_report:
            logger.info("Generating comprehensive data summary report...")
            
            report = create_data_summary_report(
                train_df, val_df, test_df, 
                save_path=args.report_path
            )
            
            logger.info(f"📊 Data summary report saved to {args.report_path}")
        
        # Print summary statistics
        logger.info("📈 Final Dataset Summary:")
        logger.info(f"  Training set: {train_df.shape[0]:,} samples, {train_df.shape[1]} features")
        logger.info(f"  Validation set: {val_df.shape[0]:,} samples, {val_df.shape[1]} features")
        logger.info(f"  Test set: {test_df.shape[0]:,} samples, {test_df.shape[1]} features")
        logger.info(f"  Total processing time: {processing_time:.2f} seconds")
        
        # Print feature information
        feature_cols = [col for col in train_df.columns if col != 'MedHouseVal']
        logger.info(f"  Feature columns ({len(feature_cols)}): {feature_cols}")
        
        # Print data quality score if available
        if hasattr(pipeline, 'data_profile') and pipeline.data_profile:
            quality_score = pipeline.data_profile.get('quality_score', 'N/A')
            logger.info(f"  Data quality score: {quality_score}/10")
        
        logger.info("🎯 Data processing pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main()) 