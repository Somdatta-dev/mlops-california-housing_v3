"""
Enhanced data manager with comprehensive data processing pipeline integration.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import pickle
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .models import CaliforniaHousingData, DataQualityReport
from .data_pipeline import CaliforniaHousingPipeline, create_pipeline_config, load_processed_data, load_scaler
from .data_utils import describe_dataset, create_data_summary_report, validate_data_consistency
from ..utils.logging_config import get_logger, LoggerMixin

logger = get_logger(__name__)


class DataManager(LoggerMixin):
    """
    Enhanced data manager with comprehensive pipeline integration.
    
    This class provides a high-level interface for all data operations,
    integrating the advanced data processing pipeline with the existing
    model training workflow.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 use_pipeline: bool = True,
                 pipeline_config: Optional[Dict] = None,
                 use_dvc: bool = False):
        """
        Initialize the enhanced data manager.
        
        Args:
            data_dir: Base data directory
            use_pipeline: Whether to use the advanced pipeline
            pipeline_config: Custom pipeline configuration
            use_dvc: Whether to enable DVC integration
        """
        self.data_dir = Path(data_dir)
        self.use_pipeline = use_pipeline
        self.use_dvc = use_dvc
        
        # Directory structure
        self.raw_data_path = self.data_dir / "raw"
        self.processed_data_path = self.data_dir / "processed"
        self.interim_data_path = self.data_dir / "interim"
        self.external_data_path = self.data_dir / "external"
        
        # Create directories
        for path in [self.raw_data_path, self.processed_data_path, 
                    self.interim_data_path, self.external_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Pipeline setup
        if use_pipeline:
            default_config = {
                'raw_data_path': str(self.raw_data_path / "california_housing.csv"),
                'processed_data_path': str(self.processed_data_path),
                'interim_data_path': str(self.interim_data_path),
                'external_data_path': str(self.external_data_path)
            }
            
            if pipeline_config:
                default_config.update(pipeline_config)
            
            self.pipeline_config = create_pipeline_config(**default_config)
            self.pipeline = CaliforniaHousingPipeline(self.pipeline_config)
        else:
            self.pipeline_config = None
            self.pipeline = None
        
        # Legacy support
        self.scaler = None
        self.feature_names = None
        self.data_profile = None
        
        self.log_info(f"DataManager initialized - Pipeline: {use_pipeline}, DVC: {use_dvc}")
    
    def download_california_housing_data(self, force_download: bool = False) -> Path:
        """
        Download California Housing dataset from sklearn.
        
        Args:
            force_download: Whether to force re-download
            
        Returns:
            Path to the downloaded data file
        """
        data_file = self.raw_data_path / "california_housing.csv"
        
        if data_file.exists() and not force_download:
            self.log_info(f"Data already exists at {data_file}")
            return data_file
        
        self.log_info("Downloading California Housing dataset from sklearn...")
        
        try:
            # Fetch the dataset
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            
            # Save to CSV
            df.to_csv(data_file, index=False)
            
            self.log_info(f"✅ Dataset downloaded successfully: {df.shape}")
            self.log_info(f"   Saved to: {data_file}")
            self.log_info(f"   Size: {data_file.stat().st_size / 1024:.1f} KB")
            
            # Initialize DVC tracking if enabled
            if self.use_dvc:
                self._init_dvc_tracking()
            
            return data_file
            
        except Exception as e:
            self.log_error(f"Failed to download dataset: {e}")
            raise
    
    def _init_dvc_tracking(self):
        """Initialize DVC tracking for the dataset."""
        try:
            import subprocess
            
            # Check if DVC is initialized
            if not (Path.cwd() / ".dvc").exists():
                subprocess.run(["dvc", "init"], check=True)
                self.log_info("DVC initialized")
            
            # Add data file to DVC tracking
            data_file = self.raw_data_path / "california_housing.csv"
            if data_file.exists():
                subprocess.run(["dvc", "add", str(data_file)], check=True)
                self.log_info(f"Added {data_file} to DVC tracking")
            
            # Configure Google Drive remote if folder ID is available
            gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID")
            if gdrive_folder_id:
                subprocess.run([
                    "dvc", "remote", "add", "-d", "gdrive",
                    f"gdrive://{gdrive_folder_id}"
                ], check=False)  # Don't fail if remote already exists
                self.log_info("DVC remote configured for Google Drive")
                
        except Exception as e:
            self.log_warning(f"DVC initialization failed: {e}")
    
    def process_data(self, force_reprocess: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Process data using the advanced pipeline or legacy method.
        
        Args:
            force_reprocess: Whether to force reprocessing even if processed data exists
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Check if processed data already exists
        if not force_reprocess and self._processed_data_exists():
            self.log_info("Loading existing processed data...")
            return self.load_processed_data()
        
        if self.use_pipeline:
            return self._process_with_pipeline()
        else:
            return self._process_legacy()
    
    def _processed_data_exists(self) -> bool:
        """Check if processed data files exist."""
        required_files = ["train.csv", "validation.csv", "test.csv"]
        return all((self.processed_data_path / f).exists() for f in required_files)
    
    def _process_with_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process data using the advanced pipeline."""
        self.log_info("🚀 Processing data with advanced pipeline...")
        
        # Ensure raw data exists
        self.download_california_housing_data()
        
        # Run pipeline
        train_df, val_df, test_df = self.pipeline.run_pipeline()
        
        # Store pipeline artifacts for compatibility
        self.scaler = load_scaler(self.processed_data_path)
        self.data_profile = getattr(self.pipeline, 'data_profile', None)
        
        # Store feature names (excluding target)
        target_col = 'MedHouseVal'
        if target_col in train_df.columns:
            self.feature_names = [col for col in train_df.columns if col != target_col]
        else:
            self.feature_names = train_df.columns.tolist()
        
        self.log_info("✅ Advanced pipeline processing completed")
        return train_df, val_df, test_df
    
    def _process_legacy(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Process data using legacy method for backward compatibility."""
        self.log_info("Processing data with legacy method...")
        
        # Load data
        df = self.load_data()
        
        # Basic validation
        quality_report = self.validate_data(df)
        self.data_profile = quality_report.data_profile
        
        # Preprocess
        train_df, test_df = self.preprocess_data(df)
        
        # Create validation split from training data
        train_features = [col for col in train_df.columns if col != 'MedHouseVal']
        X_train = train_df[train_features]
        y_train = train_df['MedHouseVal']
        
        val_size = self.pipeline_config.validation_size if self.pipeline_config else 0.2
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        train_df_split = pd.concat([X_train_split, y_train_split], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        
        self.log_info("✅ Legacy processing completed")
        return train_df_split, val_df, test_df
    
    def load_data(self) -> pd.DataFrame:
        """Load raw data with automatic download if needed."""
        data_file = self.download_california_housing_data()
        
        self.log_info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        
        self.log_info(f"Loaded dataset: {df.shape}")
        return df
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously processed data."""
        try:
            train_df, val_df, test_df = load_processed_data(str(self.processed_data_path))
            
            # Load associated artifacts
            self.scaler = load_scaler(str(self.processed_data_path))
            
            # Load feature names from config if available
            config_path = self.processed_data_path / "pipeline_config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    self.feature_names = config_data.get('feature_columns', [])
                    if 'MedHouseVal' in self.feature_names:
                        self.feature_names.remove('MedHouseVal')
            
            self.log_info(f"Loaded processed data: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}")
            return train_df, val_df, test_df
            
        except Exception as e:
            self.log_error(f"Failed to load processed data: {e}")
            # Fallback to reprocessing
            return self.process_data(force_reprocess=True)
    
    def validate_data(self, data: pd.DataFrame) -> DataQualityReport:
        """Validate data quality using pipeline validator or legacy method."""
        if self.use_pipeline and self.pipeline:
            return self.pipeline.validator.validate_data_quality(data)
        else:
            return self._validate_data_legacy(data)
    
    def _validate_data_legacy(self, data: pd.DataFrame) -> DataQualityReport:
        """Legacy data validation method."""
        total_records = len(data)
        missing_count = data.isnull().sum().sum()
        duplicates = data.duplicated().sum()
        
        issues = []
        if missing_count > 0:
            issues.append(f"Found {missing_count} missing values")
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate records")
        
        # Simple quality score
        quality_score = max(0, 10 - (missing_count / total_records) * 5 - (duplicates / total_records) * 3)
        
        return DataQualityReport(
            overall_score=quality_score,
            total_records=total_records,
            missing_values_count=int(missing_count),
            duplicate_records=int(duplicates),
            data_types={"numeric": len(data.select_dtypes(include=[np.number]).columns)},
            issues_found=issues,
            data_profile={"total_rows": total_records, "total_columns": len(data.columns)}
        )
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Legacy preprocessing method for backward compatibility."""
        # Feature engineering
        df = data.copy()
        
        # Create derived features (basic version)
        if all(col in df.columns for col in ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']):
            df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
            df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
            df['population_per_household'] = df['Population'] / df['AveOccup']
        
        # Split data
        test_size = self.pipeline_config.test_size if self.pipeline_config else 0.2
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Initialize and fit scaler
        scaler_type = self.pipeline_config.scaler_type if self.pipeline_config else "standard"
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = StandardScaler()  # Default fallback
        
        # Get feature columns (excluding target)
        feature_cols = [col for col in train_df.columns if col != 'MedHouseVal']
        self.feature_names = feature_cols
        
        # Fit scaler on training data
        self.scaler.fit(train_df[feature_cols])
        
        # Transform both datasets
        train_df[feature_cols] = self.scaler.transform(train_df[feature_cols])
        test_df[feature_cols] = self.scaler.transform(test_df[feature_cols])
        
        # Save processed data
        train_df.to_csv(self.processed_data_path / "train.csv", index=False)
        test_df.to_csv(self.processed_data_path / "test.csv", index=False)
        
        # Save scaler
        joblib.dump(self.scaler, self.processed_data_path / "scaler.joblib")
        
        self.log_info("Legacy preprocessing completed")
        return train_df, test_df
    
    def prepare_inference_data(self, data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Prepare data for model inference."""
        if isinstance(data, dict):
            # Convert dictionary to DataFrame
            df = pd.DataFrame([data])
        elif isinstance(data, np.ndarray):
            # Assume it's already in the right format
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            return data
        else:
            df = data.copy()
        
        # Ensure we have the scaler
        if self.scaler is None:
            self.scaler = load_scaler(str(self.processed_data_path))
            if self.scaler is None:
                raise ValueError("No fitted scaler available. Please process data first.")
        
        # Get feature columns
        if self.feature_names is None:
            # Try to load from config
            config_path = self.processed_data_path / "pipeline_config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    feature_columns = config_data.get('feature_columns', [])
                    self.feature_names = [col for col in feature_columns if col != 'MedHouseVal']
        
        if self.feature_names is None:
            raise ValueError("Feature names not available. Please process data first.")
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        df_features = df[self.feature_names]
        
        # Transform using fitted scaler
        scaled_data = self.scaler.transform(df_features)
        
        return scaled_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        summary = {
            'data_dir': str(self.data_dir),
            'use_pipeline': self.use_pipeline,
            'use_dvc': self.use_dvc
        }
        
        # Add processed data info if available
        if self._processed_data_exists():
            try:
                train_df, val_df, test_df = self.load_processed_data()
                summary.update({
                    'datasets': {
                        'train_shape': train_df.shape,
                        'validation_shape': val_df.shape,
                        'test_shape': test_df.shape,
                        'feature_count': len(self.feature_names) if self.feature_names else 0,
                        'total_samples': train_df.shape[0] + val_df.shape[0] + test_df.shape[0]
                    }
                })
                
                if self.feature_names:
                    summary['feature_names'] = self.feature_names
                
            except Exception as e:
                summary['data_loading_error'] = str(e)
        
        # Add data profile if available
        if self.data_profile:
            summary['data_profile'] = self.data_profile
        
        return summary
    
    def generate_data_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive data report."""
        if not self._processed_data_exists():
            self.log_info("Processed data not found, running pipeline...")
            self.process_data()
        
        # Load processed data
        train_df, val_df, test_df = self.load_processed_data()
        
        # Generate report
        report_path = save_path or str(self.processed_data_path / "comprehensive_data_report.json")
        report = create_data_summary_report(train_df, val_df, test_df, save_path=report_path)
        
        self.log_info(f"📊 Comprehensive data report generated: {report_path}")
        return report
    
    def save_data_profile(self, profile: Dict[str, Any]) -> None:
        """Save data profile to file."""
        profile_path = self.interim_data_path / "data_profile.json"
        
        import json
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
        
        self.log_info(f"Data profile saved to {profile_path}")
    
    def load_data_profile(self) -> Optional[Dict[str, Any]]:
        """Load data profile from file."""
        profile_path = self.interim_data_path / "data_profile.json"
        
        if profile_path.exists():
            import json
            with open(profile_path, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (excluding target)."""
        if self.feature_names is None:
            # Try to load from processed data
            if self._processed_data_exists():
                self.load_processed_data()
        
        return self.feature_names or []
    
    def load_scaler(self):
        """Load the fitted scaler."""
        if self.scaler is None:
            self.scaler = load_scaler(str(self.processed_data_path))
        return self.scaler 