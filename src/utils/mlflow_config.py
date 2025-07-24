"""
MLflow configuration and experiment management utilities.
"""

import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from dotenv import load_dotenv

from .logging_config import get_logger

# Load environment variables
load_dotenv()

# Setup logging
logger = get_logger(__name__)


class MLflowConfig:
    """
    MLflow configuration and experiment management.
    """
    
    def __init__(self,
                 tracking_uri: Optional[str] = None,
                 experiment_name: str = "california_housing_prediction",
                 model_registry_uri: Optional[str] = None):
        """
        Initialize MLflow configuration.
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Name of the MLflow experiment
            model_registry_uri: Model registry URI
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.experiment_name = experiment_name
        self.model_registry_uri = model_registry_uri or os.getenv("MLFLOW_MODEL_REGISTRY_URI", "./mlruns")
        
        # Setup MLflow
        self._setup_mlflow()
        
        # Initialize client
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        logger.info(f"MLflow configured - Tracking URI: {self.tracking_uri}")
        logger.info(f"Experiment: {self.experiment_name}")

    def _setup_mlflow(self):
        """Setup MLflow configuration."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    name=self.experiment_name,
                    tags={
                        "project": "california_housing_prediction",
                        "created_at": datetime.now().isoformat(),
                        "description": "ML experiment for California Housing price prediction with GPU acceleration"
                    }
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    def start_run(self, 
                  run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
            
        Returns:
            Run ID
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"
        
        # Default tags
        default_tags = {
            "created_at": datetime.now().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "gpu_available": str(self._check_gpu_availability())
        }
        
        if tags:
            default_tags.update(tags)
        
        run = mlflow.start_run(run_name=run_name, tags=default_tags)
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        
        return run.info.run_id

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import cupy
                return True
            except ImportError:
                return False

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.debug(f"Logged {len(params)} parameters to MLflow")

    def log_metrics(self, 
                   metrics: Dict[str, float],
                   step: Optional[int] = None):
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for time series metrics
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value, step=step)
            else:
                logger.warning(f"Skipping invalid metric {key}: {value}")
        
        logger.debug(f"Logged {len(metrics)} metrics to MLflow")

    def log_artifacts(self, artifacts_path: str):
        """
        Log artifacts directory to MLflow.
        
        Args:
            artifacts_path: Path to artifacts directory
        """
        if Path(artifacts_path).exists():
            mlflow.log_artifacts(artifacts_path)
            logger.info(f"Logged artifacts from {artifacts_path}")
        else:
            logger.warning(f"Artifacts path not found: {artifacts_path}")

    def log_model(self, 
                  model: Any,
                  model_name: str,
                  model_type: str = "sklearn",
                  signature: Optional[Any] = None,
                  input_example: Optional[Any] = None,
                  **kwargs):
        """
        Log model to MLflow.
        
        Args:
            model: Trained model object
            model_name: Name for the model
            model_type: Type of model (sklearn, xgboost, pytorch)
            signature: Model signature
            input_example: Example input for the model
            **kwargs: Additional arguments for model logging
        """
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs
                )
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs
                )
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    **kwargs
                )
            else:
                # Generic model logging
                mlflow.log_artifact(model, model_name)
            
            logger.info(f"Logged {model_type} model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model {model_name}: {e}")
            raise

    def register_model(self, 
                      model_uri: str,
                      model_name: str,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> Any:
        """
        Register model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model to register
            model_name: Name for registered model
            description: Optional description
            tags: Optional tags
            
        Returns:
            Registered model version
        """
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            # Update description if provided
            if description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version.version,
                    description=description
                )
            
            logger.info(f"Registered model: {model_name} version {model_version.version}")
            return model_version
            
        except Exception as e:
            logger.error(f"Error registering model {model_name}: {e}")
            raise

    def promote_model(self, 
                     model_name: str,
                     version: str,
                     stage: str = "Production"):
        """
        Promote model to a specific stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
            stage: Target stage (Staging, Production, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Promoted model {model_name} v{version} to {stage}")
            
        except Exception as e:
            logger.error(f"Error promoting model {model_name} v{version}: {e}")
            raise

    def get_best_model(self, 
                      experiment_name: Optional[str] = None,
                      metric_name: str = "rmse",
                      ascending: bool = True) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the best model from experiment based on a metric.
        
        Args:
            experiment_name: Name of experiment (uses current if None)
            metric_name: Metric to optimize for
            ascending: Whether lower values are better
            
        Returns:
            Tuple of (run_id, run_info) for best model, or None if no runs found
        """
        try:
            if experiment_name is None:
                experiment_name = self.experiment_name
            
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment not found: {experiment_name}")
                return None
            
            # Search for runs with the metric
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"metrics.{metric_name} is not null",
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"]
            )
            
            if runs.empty:
                logger.warning(f"No runs found with metric {metric_name}")
                return None
            
            best_run = runs.iloc[0]
            run_id = best_run['run_id']
            
            logger.info(f"Best model found: Run {run_id} with {metric_name}={best_run[f'metrics.{metric_name}']}")
            
            return run_id, best_run.to_dict()
            
        except Exception as e:
            logger.error(f"Error finding best model: {e}")
            return None

    def load_model(self, 
                  run_id: str,
                  model_name: str) -> Any:
        """
        Load model from MLflow.
        
        Args:
            run_id: MLflow run ID
            model_name: Name of the model artifact
            
        Returns:
            Loaded model
        """
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded model from {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name} from run {run_id}: {e}")
            raise

    def log_gpu_metrics(self, gpu_metrics: Dict[str, float]):
        """
        Log GPU-specific metrics.
        
        Args:
            gpu_metrics: Dictionary of GPU metrics
        """
        gpu_metrics_prefixed = {f"gpu_{key}": value for key, value in gpu_metrics.items()}
        self.log_metrics(gpu_metrics_prefixed)

    def log_system_info(self):
        """Log system information as parameters."""
        import platform
        import psutil
        
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_available": self._check_gpu_availability()
        }
        
        # Add GPU info if available
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            system_info["gpu_count"] = gpu_count
            
            if gpu_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode()
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                system_info["gpu_name"] = gpu_name
                system_info["gpu_memory_gb"] = round(memory_info.total / (1024**3), 2)
                
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
        
        self.log_params(system_info)

    def end_run(self, status: str = "FINISHED"):
        """
        End current MLflow run.
        
        Args:
            status: Run status (FINISHED, FAILED, KILLED)
        """
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current experiment.
        
        Returns:
            Dictionary with experiment summary
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return {"error": "Experiment not found"}
            
            runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            summary = {
                "experiment_name": self.experiment_name,
                "experiment_id": experiment.experiment_id,
                "total_runs": len(runs_df),
                "lifecycle_stage": experiment.lifecycle_stage,
                "creation_time": experiment.creation_time,
                "last_update_time": experiment.last_update_time
            }
            
            if not runs_df.empty:
                # Add metric statistics
                metric_cols = [col for col in runs_df.columns if col.startswith('metrics.')]
                for col in metric_cols:
                    metric_name = col.replace('metrics.', '')
                    values = runs_df[col].dropna()
                    if not values.empty:
                        summary[f"{metric_name}_best"] = values.min()
                        summary[f"{metric_name}_mean"] = values.mean()
                        summary[f"{metric_name}_std"] = values.std()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting experiment summary: {e}")
            return {"error": str(e)} 