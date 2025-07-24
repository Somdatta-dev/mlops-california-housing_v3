"""
Data utility functions for common data operations and transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def describe_dataset(df: pd.DataFrame, include_plots: bool = False, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate comprehensive dataset description and statistics.
    
    Args:
        df: DataFrame to describe
        include_plots: Whether to generate visualization plots
        save_path: Path to save plots if generated
        
    Returns:
        Dictionary with dataset statistics and information
    """
    logger.info(f"Generating dataset description for {df.shape} data")
    
    # Basic information
    info = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict()
    }
    
    # Missing values analysis
    missing_info = {
        'total_missing': df.isnull().sum().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = df[numeric_cols].describe().to_dict()
        
        # Correlation matrix
        correlation_matrix = df[numeric_cols].corr().to_dict()
        
        # Skewness and kurtosis
        skewness = df[numeric_cols].skew().to_dict()
        kurtosis = df[numeric_cols].kurtosis().to_dict()
    else:
        numeric_stats = {}
        correlation_matrix = {}
        skewness = {}
        kurtosis = {}
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_info = {}
    
    for col in categorical_cols:
        categorical_info[col] = {
            'unique_count': df[col].nunique(),
            'unique_values': df[col].unique().tolist()[:10],  # First 10 unique values
            'value_counts': df[col].value_counts().head().to_dict()
        }
    
    # Duplicate analysis
    duplicate_info = {
        'total_duplicates': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100
    }
    
    # Outlier analysis (for numeric columns)
    outlier_info = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_info[col] = {
            'count': outliers,
            'percentage': (outliers / len(df)) * 100,
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    # Combine all information
    description = {
        'basic_info': info,
        'missing_values': missing_info,
        'numeric_statistics': numeric_stats,
        'correlation_matrix': correlation_matrix,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'categorical_info': categorical_info,
        'duplicates': duplicate_info,
        'outliers': outlier_info
    }
    
    # Generate plots if requested
    if include_plots:
        plots_info = generate_data_plots(df, save_path)
        description['plots'] = plots_info
    
    logger.info("Dataset description completed")
    return description


def generate_data_plots(df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, str]:
    """
    Generate comprehensive data visualization plots.
    
    Args:
        df: DataFrame to visualize
        save_path: Directory to save plots
        
    Returns:
        Dictionary with plot information
    """
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    plots_info = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 1. Distribution plots for numeric columns
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            dist_plot_path = save_dir / "distributions.png"
            plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
            plots_info['distributions'] = str(dist_plot_path)
        else:
            plt.show()
        
        plt.close()
    
    # 2. Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numeric_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            corr_plot_path = save_dir / "correlation_matrix.png"
            plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
            plots_info['correlation'] = str(corr_plot_path)
        else:
            plt.show()
        
        plt.close()
    
    # 3. Box plots for outlier detection
    if len(numeric_cols) > 0:
        n_cols = min(4, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                df.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Box Plot of {col}')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            box_plot_path = save_dir / "box_plots.png"
            plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
            plots_info['box_plots'] = str(box_plot_path)
        else:
            plt.show()
        
        plt.close()
    
    # 4. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        
        if save_path:
            missing_plot_path = save_dir / "missing_values.png"
            plt.savefig(missing_plot_path, dpi=300, bbox_inches='tight')
            plots_info['missing_values'] = str(missing_plot_path)
        else:
            plt.show()
        
        plt.close()
    
    logger.info(f"Generated {len(plots_info)} visualization plots")
    return plots_info


def detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                   methods: List[str] = ['iqr', 'zscore']) -> Dict[str, Any]:
    """
    Detect outliers using multiple methods.
    
    Args:
        df: DataFrame to analyze
        columns: Columns to analyze (default: all numeric)
        methods: Detection methods to use
        
    Returns:
        Dictionary with outlier information by method and column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers_info = {}
    
    for method in methods:
        outliers_info[method] = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_mask = z_scores > 3
                
            elif method == 'modified_zscore':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                modified_z_scores = 0.6745 * (df[col] - median) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
                
            else:
                logger.warning(f"Unknown outlier detection method: {method}")
                continue
            
            outlier_indices = df[outlier_mask].index.tolist()
            
            outliers_info[method][col] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(df)) * 100,
                'indices': outlier_indices[:100],  # First 100 outlier indices
                'values': df.loc[outlier_indices[:10], col].tolist()  # First 10 outlier values
            }
    
    logger.info(f"Outlier detection completed for {len(columns)} columns using {len(methods)} methods")
    return outliers_info


def feature_selection(X: pd.DataFrame, y: pd.Series, method: str = 'f_regression', 
                     k: int = 10) -> Dict[str, Any]:
    """
    Perform feature selection using various methods.
    
    Args:
        X: Feature matrix
        y: Target variable
        method: Selection method ('f_regression', 'mutual_info', 'correlation')
        k: Number of features to select
        
    Returns:
        Dictionary with selected features and scores
    """
    logger.info(f"Performing feature selection using {method}, selecting top {k} features")
    
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        scores = selector.scores_
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        scores = selector.scores_
        selected_features = X.columns[selector.get_support()].tolist()
        
    elif method == 'correlation':
        correlations = X.corrwith(y).abs()
        selected_features = correlations.nlargest(k).index.tolist()
        scores = correlations[selected_features].values
        X_selected = X[selected_features].values
        
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    # Create feature importance ranking
    feature_scores = dict(zip(selected_features, scores))
    ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    result = {
        'method': method,
        'selected_features': selected_features,
        'feature_scores': feature_scores,
        'ranked_features': ranked_features,
        'X_selected_shape': X_selected.shape,
        'original_shape': X.shape
    }
    
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    return result


def validate_data_consistency(df1: pd.DataFrame, df2: pd.DataFrame, 
                            name1: str = "Dataset 1", name2: str = "Dataset 2") -> Dict[str, Any]:
    """
    Validate consistency between two datasets (e.g., train vs test).
    
    Args:
        df1: First dataset
        df2: Second dataset
        name1: Name for first dataset
        name2: Name for second dataset
        
    Returns:
        Dictionary with consistency check results
    """
    logger.info(f"Validating consistency between {name1} and {name2}")
    
    consistency_report = {
        'shapes': {name1: df1.shape, name2: df2.shape},
        'columns_match': set(df1.columns) == set(df2.columns),
        'column_differences': {
            'only_in_df1': list(set(df1.columns) - set(df2.columns)),
            'only_in_df2': list(set(df2.columns) - set(df1.columns))
        },
        'dtype_consistency': {},
        'distribution_differences': {}
    }
    
    # Check data types
    common_columns = set(df1.columns) & set(df2.columns)
    
    for col in common_columns:
        df1_dtype = str(df1[col].dtype)
        df2_dtype = str(df2[col].dtype)
        consistency_report['dtype_consistency'][col] = {
            'df1_dtype': df1_dtype,
            'df2_dtype': df2_dtype,
            'match': df1_dtype == df2_dtype
        }
    
    # Check distributions for numeric columns
    numeric_cols = df1.select_dtypes(include=[np.number]).columns
    common_numeric = set(numeric_cols) & set(df2.select_dtypes(include=[np.number]).columns)
    
    for col in common_numeric:
        if col in common_columns:
            # Kolmogorov-Smirnov test for distribution similarity
            try:
                ks_stat, p_value = stats.ks_2samp(df1[col].dropna(), df2[col].dropna())
                consistency_report['distribution_differences'][col] = {
                    'ks_statistic': float(ks_stat),
                    'p_value': float(p_value),
                    'distributions_similar': p_value > 0.05  # α = 0.05
                }
            except Exception as e:
                consistency_report['distribution_differences'][col] = {
                    'error': str(e)
                }
    
    # Overall consistency score
    scores = []
    scores.append(1.0 if consistency_report['columns_match'] else 0.0)
    
    dtype_matches = [info['match'] for info in consistency_report['dtype_consistency'].values()]
    if dtype_matches:
        scores.append(sum(dtype_matches) / len(dtype_matches))
    
    dist_similar = [info.get('distributions_similar', False) 
                   for info in consistency_report['distribution_differences'].values()]
    if dist_similar:
        scores.append(sum(dist_similar) / len(dist_similar))
    
    consistency_report['overall_consistency_score'] = sum(scores) / len(scores) if scores else 0.0
    
    logger.info(f"Consistency validation completed. Score: {consistency_report['overall_consistency_score']:.2f}")
    return consistency_report


def create_data_summary_report(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                             test_df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive data summary report for train/val/test splits.
    
    Args:
        train_df: Training dataset
        val_df: Validation dataset  
        test_df: Test dataset
        save_path: Path to save the report
        
    Returns:
        Comprehensive data summary report
    """
    logger.info("Creating comprehensive data summary report")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'datasets': {
            'train': describe_dataset(train_df),
            'validation': describe_dataset(val_df),
            'test': describe_dataset(test_df)
        },
        'consistency_checks': {
            'train_vs_validation': validate_data_consistency(train_df, val_df, "Train", "Validation"),
            'train_vs_test': validate_data_consistency(train_df, test_df, "Train", "Test"),
            'validation_vs_test': validate_data_consistency(val_df, test_df, "Validation", "Test")
        }
    }
    
    # Add feature analysis if target column exists
    target_col = 'MedHouseVal'
    if target_col in train_df.columns:
        feature_cols = [col for col in train_df.columns if col != target_col]
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        
        # Feature importance analysis
        try:
            feature_analysis = feature_selection(X_train, y_train, method='f_regression', k=10)
            report['feature_analysis'] = feature_analysis
        except Exception as e:
            logger.warning(f"Feature analysis failed: {e}")
            report['feature_analysis'] = {'error': str(e)}
    
    # Save report if path provided
    if save_path:
        report_path = Path(save_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Data summary report saved to {report_path}")
    
    logger.info("Data summary report completed")
    return report


def load_and_validate_data(data_path: str, expected_columns: Optional[List[str]] = None,
                          expected_shape: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """
    Load data with validation checks.
    
    Args:
        data_path: Path to data file
        expected_columns: Expected column names
        expected_shape: Expected shape (rows, cols)
        
    Returns:
        Validated DataFrame
    """
    logger.info(f"Loading and validating data from {data_path}")
    
    # Load data
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if data_path.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    elif data_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix.lower() == '.json':
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Validation checks
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)
        
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
    
    if expected_shape:
        if df.shape != expected_shape:
            logger.warning(f"Shape mismatch: expected {expected_shape}, got {df.shape}")
    
    logger.info(f"Successfully loaded and validated data: {df.shape}")
    return df 