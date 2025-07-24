"""
Comprehensive data processing pipeline for California Housing dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import yaml
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import warnings

from .models import CaliforniaHousingData, DataQualityReport
from ..utils.logging_config import get_logger, LoggerMixin

logger = get_logger(__name__)


@dataclass
class DataPipelineConfig:
    """Configuration for data processing pipeline."""
    # Data sources
    raw_data_path: str = "data/raw/california_housing.csv"
    processed_data_path: str = "data/processed"
    interim_data_path: str = "data/interim"
    external_data_path: str = "data/external"
    
    # Processing parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    enable_feature_engineering: bool = True
    enable_outlier_detection: bool = True
    enable_feature_selection: bool = False
    
    # Scaling
    scaler_type: str = "standard"  # standard, robust, minmax
    
    # Data quality
    missing_threshold: float = 0.1  # Max allowed missing values ratio
    outlier_threshold: float = 3.0  # Z-score threshold for outliers
    
    # Validation
    enable_data_validation: bool = True
    save_data_profile: bool = True


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get processor parameters."""
        pass


class OutlierDetector(DataProcessor):
    """Detect and handle outliers in the dataset."""
    
    def __init__(self, method: str = "zscore", threshold: float = 3.0, action: str = "cap"):
        """
        Initialize outlier detector.
        
        Args:
            method: Detection method ('zscore', 'iqr', 'isolation_forest')
            threshold: Threshold for outlier detection
            action: Action to take ('remove', 'cap', 'log')
        """
        self.method = method
        self.threshold = threshold
        self.action = action
        self.outlier_bounds_ = {}
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'MedHouseVal':  # Skip target variable
                continue
                
            if self.method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > self.threshold
            elif self.method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                self.outlier_bounds_[col] = (lower_bound, upper_bound)
            else:
                continue
                
            if self.action == "remove":
                df = df[~outliers]
            elif self.action == "cap":
                if self.method == "zscore":
                    # Cap to 3 standard deviations
                    mean, std = df[col].mean(), df[col].std()
                    df[col] = np.clip(df[col], mean - self.threshold * std, mean + self.threshold * std)
                elif self.method == "iqr":
                    lower_bound, upper_bound = self.outlier_bounds_[col]
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
            elif self.action == "log":
                # Apply log transformation to reduce impact
                df[col] = np.log1p(np.abs(df[col])) * np.sign(df[col])
        
        return df
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "action": self.action,
            "outlier_bounds": self.outlier_bounds_
        }


class FeatureEngineer(DataProcessor):
    """Create engineered features from raw data."""
    
    def __init__(self, enable_interactions: bool = True, enable_polynomials: bool = False):
        """
        Initialize feature engineer.
        
        Args:
            enable_interactions: Whether to create interaction features
            enable_polynomials: Whether to create polynomial features
        """
        self.enable_interactions = enable_interactions
        self.enable_polynomials = enable_polynomials
        self.feature_names_ = []
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features."""
        df = data.copy()
        
        # Original California Housing derived features
        if all(col in df.columns for col in ['AveRooms', 'AveBedrms', 'Population', 'AveOccup']):
            # Rooms per household
            df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
            
            # Bedrooms per room ratio
            df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
            
            # Population per household
            df['population_per_household'] = df['Population'] / df['AveOccup']
            
            self.feature_names_.extend([
                'rooms_per_household', 'bedrooms_per_room', 'population_per_household'
            ])
        
        # Geographic features
        if all(col in df.columns for col in ['Latitude', 'Longitude']):
            # Distance from major cities (approximate)
            # San Francisco: 37.7749, -122.4194
            df['distance_to_sf'] = np.sqrt(
                (df['Latitude'] - 37.7749)**2 + (df['Longitude'] + 122.4194)**2
            )
            
            # Los Angeles: 34.0522, -118.2437
            df['distance_to_la'] = np.sqrt(
                (df['Latitude'] - 34.0522)**2 + (df['Longitude'] + 118.2437)**2
            )
            
            # Coastal proximity (rough estimate)
            df['coastal_proximity'] = np.where(df['Longitude'] > -121.0, 1, 0)
            
            self.feature_names_.extend([
                'distance_to_sf', 'distance_to_la', 'coastal_proximity'
            ])
        
        # Income-based features
        if 'MedInc' in df.columns:
            # Income categories
            df['income_category'] = pd.cut(
                df['MedInc'], 
                bins=[0, 2.5, 4.5, 6.0, float('inf')], 
                labels=['low', 'medium', 'high', 'very_high']
            )
            
            # Convert to dummy variables
            income_dummies = pd.get_dummies(df['income_category'], prefix='income')
            df = pd.concat([df, income_dummies], axis=1)
            df.drop('income_category', axis=1, inplace=True)
            
            self.feature_names_.extend(income_dummies.columns.tolist())
        
        # Age-based features
        if 'HouseAge' in df.columns:
            # Age categories
            df['age_category'] = pd.cut(
                df['HouseAge'],
                bins=[0, 10, 25, 40, float('inf')],
                labels=['new', 'recent', 'mature', 'old']
            )
            
            # Convert to dummy variables
            age_dummies = pd.get_dummies(df['age_category'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)
            df.drop('age_category', axis=1, inplace=True)
            
            self.feature_names_.extend(age_dummies.columns.tolist())
        
        # Interaction features
        if self.enable_interactions:
            if all(col in df.columns for col in ['MedInc', 'AveRooms']):
                df['income_rooms_interaction'] = df['MedInc'] * df['AveRooms']
                self.feature_names_.append('income_rooms_interaction')
            
            if all(col in df.columns for col in ['MedInc', 'HouseAge']):
                df['income_age_interaction'] = df['MedInc'] * df['HouseAge']
                self.feature_names_.append('income_age_interaction')
        
        # Polynomial features (if enabled)
        if self.enable_polynomials:
            numeric_cols = ['MedInc', 'AveRooms', 'Population']
            for col in numeric_cols:
                if col in df.columns:
                    df[f'{col}_squared'] = df[col] ** 2
                    self.feature_names_.append(f'{col}_squared')
        
        return df
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "enable_interactions": self.enable_interactions,
            "enable_polynomials": self.enable_polynomials,
            "feature_names": self.feature_names_
        }


class DataValidator(LoggerMixin):
    """Validate data quality and consistency."""
    
    def __init__(self, config: DataPipelineConfig):
        self.config = config
        
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """Validate overall data quality."""
        self.log_info("Validating data quality...")
        
        # Basic statistics
        total_rows = len(data)
        total_cols = len(data.columns)
        
        # Missing values
        missing_counts = data.isnull().sum()
        missing_ratios = missing_counts / total_rows
        
        # Duplicate rows
        duplicates = data.duplicated().sum()
        
        # Data types
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Outliers (using IQR method)
        outlier_counts = {}
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
        
        # Data ranges
        data_ranges = {}
        for col in numeric_cols:
            data_ranges[col] = {
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'mean': float(data[col].mean()),
                'std': float(data[col].std())
            }
        
        # Quality score calculation
        missing_penalty = min(missing_ratios.max() * 10, 5)  # Max 5 points penalty
        duplicate_penalty = min((duplicates / total_rows) * 10, 3)  # Max 3 points penalty
        outlier_penalty = min(sum(outlier_counts.values()) / total_rows * 5, 2)  # Max 2 points penalty
        
        quality_score = max(0, 10 - missing_penalty - duplicate_penalty - outlier_penalty)
        
        # Create data profile
        data_profile = {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_values': missing_counts.to_dict(),
            'missing_ratios': missing_ratios.to_dict(),
            'duplicates': int(duplicates),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'outlier_counts': outlier_counts,
            'data_ranges': data_ranges,
            'quality_score': quality_score
        }
        
        # Issues found
        issues = []
        
        # Check for high missing values
        high_missing = missing_ratios[missing_ratios > self.config.missing_threshold]
        if not high_missing.empty:
            issues.append(f"High missing values in columns: {high_missing.index.tolist()}")
        
        # Check for duplicates
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")
        
        # Check for excessive outliers
        high_outliers = {k: v for k, v in outlier_counts.items() if v / total_rows > 0.05}
        if high_outliers:
            issues.append(f"High outlier counts in columns: {list(high_outliers.keys())}")
        
        self.log_info(f"Data quality validation completed. Score: {quality_score:.2f}/10")
        
        return DataQualityReport(
            overall_score=quality_score,
            total_records=total_rows,
            missing_values_count=int(missing_counts.sum()),
            duplicate_records=int(duplicates),
            data_types={"numeric": len(numeric_cols), "categorical": len(categorical_cols)},
            issues_found=issues,
            data_profile=data_profile
        )
    
    def validate_schema(self, data: pd.DataFrame, expected_columns: List[str]) -> bool:
        """Validate data schema matches expectations."""
        missing_cols = set(expected_columns) - set(data.columns)
        extra_cols = set(data.columns) - set(expected_columns)
        
        if missing_cols:
            self.log_error(f"Missing required columns: {missing_cols}")
            return False
        
        if extra_cols:
            self.log_warning(f"Extra columns found: {extra_cols}")
        
        return True
    
    def validate_target_distribution(self, data: pd.DataFrame, target_col: str = 'MedHouseVal') -> Dict[str, Any]:
        """Validate target variable distribution."""
        if target_col not in data.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        target = data[target_col]
        
        stats = {
            'count': len(target),
            'mean': float(target.mean()),
            'std': float(target.std()),
            'min': float(target.min()),
            'max': float(target.max()),
            'skewness': float(target.skew()),
            'kurtosis': float(target.kurtosis()),
            'missing_count': int(target.isnull().sum())
        }
        
        # Check for reasonable distribution
        warnings = []
        if stats['skewness'] > 2:
            warnings.append("Target variable is highly right-skewed")
        elif stats['skewness'] < -2:
            warnings.append("Target variable is highly left-skewed")
        
        if stats['missing_count'] > 0:
            warnings.append(f"Target variable has {stats['missing_count']} missing values")
        
        stats['warnings'] = warnings
        return stats


class CaliforniaHousingPipeline(LoggerMixin):
    """Complete data processing pipeline for California Housing dataset."""
    
    def __init__(self, config: DataPipelineConfig):
        """
        Initialize the data pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.processors = []
        self.scaler = None
        self.validator = DataValidator(config)
        self.data_profile = None
        self.processing_stats = {}
        
        # Initialize processors based on config
        if config.enable_outlier_detection:
            self.processors.append(
                OutlierDetector(
                    method="iqr", 
                    threshold=config.outlier_threshold, 
                    action="cap"
                )
            )
        
        if config.enable_feature_engineering:
            self.processors.append(
                FeatureEngineer(enable_interactions=True, enable_polynomials=False)
            )
        
        # Initialize scaler
        if config.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif config.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif config.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {config.scaler_type}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from source."""
        self.log_info(f"Loading raw data from {self.config.raw_data_path}")
        
        data_path = Path(self.config.raw_data_path)
        
        if not data_path.exists():
            # Download California Housing data if not exists
            self.log_info("Raw data not found, downloading from sklearn...")
            from sklearn.datasets import fetch_california_housing
            
            housing = fetch_california_housing(as_frame=True)
            df = housing.frame
            
            # Ensure directory exists
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            df.to_csv(data_path, index=False)
            self.log_info(f"Downloaded and saved data to {data_path}")
        else:
            # Load existing data
            df = pd.read_csv(data_path)
            self.log_info(f"Loaded {len(df)} records from {data_path}")
        
        return df
    
    def validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data."""
        self.log_info("Validating and cleaning data...")
        
        # Validate data quality
        if self.config.enable_data_validation:
            quality_report = self.validator.validate_data_quality(data)
            self.data_profile = quality_report.data_profile
            
            # Save data profile if configured
            if self.config.save_data_profile:
                profile_path = Path(self.config.interim_data_path) / "data_profile.json"
                profile_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(profile_path, 'w') as f:
                    json.dump(quality_report.data_profile, f, indent=2, default=str)
                
                self.log_info(f"Data profile saved to {profile_path}")
            
            # Log quality issues
            if quality_report.issues_found:
                for issue in quality_report.issues_found:
                    self.log_warning(f"Data quality issue: {issue}")
        
        # Basic cleaning
        df = data.copy()
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_duplicates = initial_rows - len(df)
        
        if removed_duplicates > 0:
            self.log_info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values (simple imputation for now)
        missing_cols = df.columns[df.isnull().any()]
        if not missing_cols.empty:
            self.log_info(f"Imputing missing values in columns: {missing_cols.tolist()}")
            
            # Use median for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
        
        self.processing_stats['cleaning'] = {
            'removed_duplicates': removed_duplicates,
            'imputed_columns': missing_cols.tolist()
        }
        
        return df
    
    def apply_processors(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured processors to the data."""
        df = data.copy()
        
        self.log_info(f"Applying {len(self.processors)} data processors...")
        
        processor_stats = {}
        
        for i, processor in enumerate(self.processors):
            processor_name = processor.__class__.__name__
            self.log_info(f"Applying {processor_name}...")
            
            initial_shape = df.shape
            df = processor.process(df)
            final_shape = df.shape
            
            processor_stats[processor_name] = {
                'initial_shape': initial_shape,
                'final_shape': final_shape,
                'params': processor.get_params()
            }
            
            self.log_info(f"{processor_name} completed: {initial_shape} -> {final_shape}")
        
        self.processing_stats['processors'] = processor_stats
        return df
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        self.log_info("Splitting data into train/validation/test sets...")
        
        # Separate features and target
        if 'MedHouseVal' in data.columns:
            X = data.drop('MedHouseVal', axis=1)
            y = data['MedHouseVal']
        else:
            # If target column has different name, assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=None  # For regression
        )
        
        # Second split: train vs validation
        validation_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size_adjusted,
            random_state=self.config.random_state
        )
        
        # Combine back into DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        self.log_info(f"Data split completed:")
        self.log_info(f"  Train: {train_df.shape}")
        self.log_info(f"  Validation: {val_df.shape}")
        self.log_info(f"  Test: {test_df.shape}")
        
        self.processing_stats['data_split'] = {
            'train_shape': train_df.shape,
            'val_shape': val_df.shape,
            'test_shape': test_df.shape,
            'test_size': self.config.test_size,
            'validation_size': self.config.validation_size
        }
        
        return train_df, val_df, test_df
    
    def fit_scaler(self, train_data: pd.DataFrame) -> None:
        """Fit the scaler on training data."""
        # Get numeric columns (excluding target)
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        if 'MedHouseVal' in numeric_cols:
            numeric_cols = numeric_cols.drop('MedHouseVal')
        
        if len(numeric_cols) > 0:
            self.log_info(f"Fitting {self.config.scaler_type} scaler on {len(numeric_cols)} features")
            self.scaler.fit(train_data[numeric_cols])
        else:
            self.log_warning("No numeric features found for scaling")
    
    def transform_data(self, data: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        df = data.copy()
        
        # Get numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'MedHouseVal' in numeric_cols:
            numeric_cols = numeric_cols.drop('MedHouseVal')
        
        if len(numeric_cols) > 0 and self.scaler is not None:
            if fit_scaler:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
                self.log_info(f"Fitted and transformed {len(numeric_cols)} features")
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
                self.log_info(f"Transformed {len(numeric_cols)} features")
        
        return df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Save processed data to files."""
        processed_dir = Path(self.config.processed_data_path)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        train_path = processed_dir / "train.csv"
        val_path = processed_dir / "validation.csv"
        test_path = processed_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        self.log_info(f"Saved processed data:")
        self.log_info(f"  Train: {train_path}")
        self.log_info(f"  Validation: {val_path}")
        self.log_info(f"  Test: {test_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = processed_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            self.log_info(f"  Scaler: {scaler_path}")
        
        # Save processing configuration and stats
        config_path = processed_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            config_dict = {
                'config': self.config.__dict__,
                'processing_stats': self.processing_stats,
                'feature_columns': list(train_df.columns)
            }
            json.dump(config_dict, f, indent=2, default=str)
        
        self.log_info(f"  Config: {config_path}")
    
    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run the complete data processing pipeline."""
        self.log_info("🚀 Starting California Housing data processing pipeline...")
        
        try:
            # Step 1: Load raw data
            raw_data = self.load_raw_data()
            
            # Step 2: Validate and clean
            cleaned_data = self.validate_and_clean(raw_data)
            
            # Step 3: Apply processors (outlier detection, feature engineering)
            processed_data = self.apply_processors(cleaned_data)
            
            # Step 4: Split data
            train_df, val_df, test_df = self.split_data(processed_data)
            
            # Step 5: Fit scaler on training data
            self.fit_scaler(train_df)
            
            # Step 6: Transform all datasets
            train_df = self.transform_data(train_df, fit_scaler=False)  # Already fitted
            val_df = self.transform_data(val_df, fit_scaler=False)
            test_df = self.transform_data(test_df, fit_scaler=False)
            
            # Step 7: Save processed data
            self.save_processed_data(train_df, val_df, test_df)
            
            self.log_info("✅ Data processing pipeline completed successfully!")
            
            # Log final statistics
            self.log_info("📊 Final dataset statistics:")
            self.log_info(f"  Training set: {train_df.shape[0]} samples, {train_df.shape[1]} features")
            self.log_info(f"  Validation set: {val_df.shape[0]} samples, {val_df.shape[1]} features")
            self.log_info(f"  Test set: {test_df.shape[0]} samples, {test_df.shape[1]} features")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            self.log_error(f"Pipeline failed: {str(e)}")
            raise


def create_pipeline_config(**kwargs) -> DataPipelineConfig:
    """Create a data pipeline configuration with optional overrides."""
    config = DataPipelineConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
    
    return config


def load_processed_data(processed_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously processed data."""
    processed_path = Path(processed_dir)
    
    train_df = pd.read_csv(processed_path / "train.csv")
    val_df = pd.read_csv(processed_path / "validation.csv")
    test_df = pd.read_csv(processed_path / "test.csv")
    
    return train_df, val_df, test_df


def load_scaler(processed_dir: str = "data/processed"):
    """Load the fitted scaler."""
    scaler_path = Path(processed_dir) / "scaler.joblib"
    
    if scaler_path.exists():
        return joblib.load(scaler_path)
    else:
        logger.warning(f"Scaler not found at {scaler_path}")
        return None 