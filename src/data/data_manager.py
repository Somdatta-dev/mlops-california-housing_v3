"""
Data management system with DVC integration and comprehensive validation.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import dvc.api
from dotenv import load_dotenv
import pickle
import json
from datetime import datetime

from .models import CaliforniaHousingData, DataQualityReport

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


class DataManager:
    """
    Comprehensive data management with DVC integration, validation, and preprocessing.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 dvc_remote: Optional[str] = None,
                 use_dvc: bool = True):
        """
        Initialize DataManager.
        
        Args:
            data_dir: Base directory for data storage
            dvc_remote: DVC remote storage configuration
            use_dvc: Whether to use DVC for data versioning
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.interim_dir = self.data_dir / "interim"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.interim_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.use_dvc = use_dvc
        self.dvc_remote = dvc_remote or os.getenv("DVC_REMOTE", "gdrive")
        
        # Data paths
        self.raw_data_path = self.raw_dir / "california_housing.csv"
        self.processed_data_path = self.processed_dir / "california_housing_processed.csv"
        self.train_data_path = self.processed_dir / "train.csv"
        self.test_data_path = self.processed_dir / "test.csv"
        self.scaler_path = self.processed_dir / "scaler.pkl"
        self.data_profile_path = self.processed_dir / "data_profile.json"
        
        logger.info(f"DataManager initialized with data_dir: {self.data_dir}")

    def download_california_housing_data(self) -> pd.DataFrame:
        """
        Download California Housing dataset from sklearn.
        
        Returns:
            DataFrame containing the California Housing data
        """
        logger.info("Downloading California Housing dataset from sklearn")
        
        try:
            # Fetch data from sklearn
            housing = fetch_california_housing(as_frame=True)
            
            # Combine features and target
            df = housing.frame
            
            # Rename columns to match our Pydantic model
            column_mapping = {
                'MedInc': 'MedInc',
                'HouseAge': 'HouseAge', 
                'AveRooms': 'AveRooms',
                'AveBedrms': 'AveBedrms',
                'Population': 'Population',
                'AveOccup': 'AveOccup',
                'Latitude': 'Latitude',
                'Longitude': 'Longitude',
                'MedHouseVal': 'target'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Save raw data
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Raw data saved to {self.raw_data_path}")
            
            # Initialize DVC tracking if enabled
            if self.use_dvc:
                self._init_dvc_tracking()
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading California Housing data: {e}")
            raise

    def _init_dvc_tracking(self):
        """Initialize DVC tracking for the dataset."""
        try:
            import subprocess
            
            # Initialize DVC if not already done
            if not (Path.cwd() / ".dvc").exists():
                subprocess.run(["dvc", "init"], check=True)
                logger.info("DVC initialized")
            
            # Add data to DVC tracking
            if self.raw_data_path.exists():
                subprocess.run(["dvc", "add", str(self.raw_data_path)], check=True)
                logger.info(f"Added {self.raw_data_path} to DVC tracking")
            
            # Configure remote if specified
            gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID")
            if gdrive_folder_id:
                subprocess.run([
                    "dvc", "remote", "add", "-d", "gdrive", 
                    f"gdrive://{gdrive_folder_id}"
                ], check=False)  # Don't fail if remote already exists
                logger.info("DVC remote configured for Google Drive")
                
        except Exception as e:
            logger.warning(f"DVC initialization failed: {e}")

    def load_data(self, reload: bool = False) -> pd.DataFrame:
        """
        Load data, downloading if necessary.
        
        Args:
            reload: Force redownload of data
            
        Returns:
            DataFrame containing the data
        """
        if reload or not self.raw_data_path.exists():
            return self.download_california_housing_data()
        else:
            logger.info(f"Loading existing data from {self.raw_data_path}")
            return pd.read_csv(self.raw_data_path)

    def validate_data(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Validate data using Pydantic models and generate quality report.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataQualityReport with validation results
        """
        logger.info("Validating data quality")
        
        validation_errors = []
        valid_records = 0
        
        for idx, row in df.iterrows():
            try:
                # Convert row to dict and validate with Pydantic
                row_dict = row.to_dict()
                CaliforniaHousingData(**row_dict)
                valid_records += 1
            except Exception as e:
                validation_errors.append({
                    "record_id": idx,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Generate data profile
        data_profile = self._generate_data_profile(df)
        
        # Create quality report
        report = DataQualityReport(
            total_records=len(df),
            valid_records=valid_records,
            invalid_records=len(df) - valid_records,
            validation_errors=validation_errors[:100],  # Limit errors shown
            data_profile=data_profile
        )
        
        logger.info(f"Data validation completed. Quality score: {report.quality_score:.2f}%")
        return report

    def _generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic data profiling statistics."""
        profile = {
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "numeric_summary": {}
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile["numeric_summary"][col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q50": float(df[col].quantile(0.50)),
                "q75": float(df[col].quantile(0.75))
            }
        
        return profile

    def preprocess_data(self, 
                       df: pd.DataFrame,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       scaling_method: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess data with feature engineering and scaling.
        
        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            scaling_method: Scaling method ('standard', 'robust', or 'none')
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info("Preprocessing data")
        
        # Create feature engineering
        df_processed = df.copy()
        
        # Feature engineering
        df_processed['rooms_per_household'] = df_processed['AveRooms']
        df_processed['bedrooms_per_room'] = df_processed['AveBedrms'] / df_processed['AveRooms']
        df_processed['population_per_household'] = df_processed['Population'] / df_processed['AveOccup']
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
        df_processed = df_processed.fillna(df_processed.mean())
        
        # Split features and target
        feature_cols = [col for col in df_processed.columns if col != 'target']
        X = df_processed[feature_cols]
        y = df_processed['target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Apply scaling
        scaler = None
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        
        if scaler:
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Save scaler
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Scaler saved to {self.scaler_path}")
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Combine with target
        train_df = pd.concat([X_train_scaled, y_train], axis=1)
        test_df = pd.concat([X_test_scaled, y_test], axis=1)
        
        # Save processed data
        train_df.to_csv(self.train_data_path, index=False)
        test_df.to_csv(self.test_data_path, index=False)
        
        logger.info(f"Processed data saved to {self.processed_dir}")
        logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        
        return train_df, test_df

    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load preprocessed train and test data.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if not (self.train_data_path.exists() and self.test_data_path.exists()):
            raise FileNotFoundError("Processed data not found. Run preprocess_data first.")
        
        train_df = pd.read_csv(self.train_data_path)
        test_df = pd.read_csv(self.test_data_path)
        
        logger.info("Loaded processed data")
        return train_df, test_df

    def load_scaler(self):
        """Load the saved scaler."""
        if not self.scaler_path.exists():
            raise FileNotFoundError("Scaler not found. Run preprocess_data first.")
        
        with open(self.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return scaler

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        if self.train_data_path.exists():
            df = pd.read_csv(self.train_data_path)
            return [col for col in df.columns if col != 'target']
        else:
            # Default feature names
            return [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                'Population', 'AveOccup', 'Latitude', 'Longitude',
                'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
            ]

    def save_data_profile(self, profile: Dict[str, Any]):
        """Save data profile to JSON file."""
        with open(self.data_profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        logger.info(f"Data profile saved to {self.data_profile_path}")

    def load_data_profile(self) -> Optional[Dict[str, Any]]:
        """Load data profile from JSON file."""
        if self.data_profile_path.exists():
            with open(self.data_profile_path, 'r') as f:
                return json.load(f)
        return None

    def prepare_inference_data(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare single record for inference.
        
        Args:
            data: Dictionary with feature values
            
        Returns:
            Preprocessed feature array ready for model inference
        """
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Apply same feature engineering as training
        df['rooms_per_household'] = df['AveRooms']
        df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
        df['population_per_household'] = df['Population'] / df['AveOccup']
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)  # Use 0 for inference, as we can't use mean
        
        # Get feature columns in correct order
        feature_cols = self.get_feature_names()
        
        # Ensure all features are present
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Select and order features
        df = df[feature_cols]
        
        # Apply scaling if scaler exists
        try:
            scaler = self.load_scaler()
            df_scaled = scaler.transform(df)
            return df_scaled
        except FileNotFoundError:
            logger.warning("No scaler found, returning unscaled data")
            return df.values

    def get_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive data summary."""
        summary = {
            "data_paths": {
                "raw": str(self.raw_data_path),
                "processed": str(self.processed_data_path),
                "train": str(self.train_data_path),
                "test": str(self.test_data_path)
            },
            "files_exist": {
                "raw": self.raw_data_path.exists(),
                "train": self.train_data_path.exists(),
                "test": self.test_data_path.exists(),
                "scaler": self.scaler_path.exists()
            }
        }
        
        # Add data shapes if files exist
        if self.train_data_path.exists():
            train_df = pd.read_csv(self.train_data_path)
            summary["train_shape"] = train_df.shape
        
        if self.test_data_path.exists():
            test_df = pd.read_csv(self.test_data_path)
            summary["test_shape"] = test_df.shape
        
        return summary 