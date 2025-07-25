import pytest
import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.model_comparison import ModelComparator, plot_comparison
from src.utils.mlflow_config import MLflowConfig

# Use an in-memory SQLite database for testing to avoid file locking issues
MLFLOW_TRACKING_URI = "sqlite:///:memory:"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@pytest.fixture(scope="module")
def mlflow_setup():
    """Setup MLflow experiment for the test module."""
    experiment_name = "model_comparison_test"
    mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    yield exp.experiment_id
    # No teardown needed for in-memory database

@pytest.fixture(scope="module")
def sample_data():
    """Create sample data for testing."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    y = pd.Series(np.random.rand(100) * 10)
    return X, y

@pytest.fixture(scope="module")
def registered_models(mlflow_setup, sample_data):
    """Register dummy models for testing."""
    X, y = sample_data
    model_names = ["linear_model", "random_forest_dummy"]
    versions = []

    for name in model_names:
        with mlflow.start_run():
            model = LinearRegression()
            model.fit(X, y)
            mlflow.sklearn.log_model(model, name, registered_model_name=name)
            client = mlflow.tracking.MlflowClient()
            result = client.get_latest_versions(name, stages=["None"])
            versions.append(result[0].version)
        mlflow.end_run()

    return model_names, [int(v) for v in versions]


class TestModelComparator:
    def test_initialization(self, registered_models):
        """Test ModelComparator initialization."""
        model_names, model_versions = registered_models
        comparator = ModelComparator(model_names=model_names, model_versions=model_versions)
        assert comparator.cv.get_n_splits() == 5
        assert comparator.metrics == ["rmse", "mae", "r2"]

    def test_evaluation(self, registered_models, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        model_names, model_versions = registered_models
        comparator = ModelComparator(model_names=model_names, model_versions=model_versions)
        results = comparator.evaluate(X, y)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(model_names)
        assert "rmse_mean" in results.columns
        assert "r2_std" in results.columns

    def test_select_best(self, registered_models, sample_data):
        """Test best model selection."""
        X, y = sample_data
        model_names, model_versions = registered_models
        comparator = ModelComparator(model_names=model_names, model_versions=model_versions)
        results = comparator.evaluate(X, y)
        best_model, scores = comparator.select_best(results, priority=["rmse", "mae"])

        assert isinstance(best_model, str)
        assert ":" in best_model
        assert "rmse" in scores
        assert "mae" in scores

    def test_statistical_tests(self, registered_models, sample_data):
        """Test statistical significance testing."""
        X, y = sample_data
        model_names, model_versions = registered_models
        comparator = ModelComparator(model_names=model_names, model_versions=model_versions)
        results = comparator.evaluate(X, y)
        
        # This test passes if it runs without the NameError
        stats = comparator.statistical_tests(X, y, results, metric="rmse")
        assert "best_model" in stats
        assert "p_values" in stats
        assert len(stats["p_values"]) == len(model_names) - 1

    def test_register_best(self, registered_models, sample_data):
        """Test registering the best model."""
        X, y = sample_data
        model_names, model_versions = registered_models
        comparator = ModelComparator(model_names=model_names, model_versions=model_versions)
        results = comparator.evaluate(X, y)
        best_model, _ = comparator.select_best(results)

        dest_name = "best_california_model"
        comparator.register_best(best_model, dest_name, stage="Staging")

        client = mlflow.tracking.MlflowClient()
        latest = client.get_latest_versions(dest_name, stages=["Staging"])
        assert len(latest) > 0
        assert latest[0].name == dest_name

def test_plot_comparison(tmp_path):
    """Test the comparison plotting utility."""
    data = {
        "model": ["model_a:1", "model_b:1"],
        "rmse_mean": [0.5, 0.6], "rmse_std": [0.01, 0.02],
        "mae_mean": [0.4, 0.5], "mae_std": [0.01, 0.02],
        "r2_mean": [0.9, 0.8], "r2_std": [0.01, 0.02],
    }
    results = pd.DataFrame(data)
    output_file = tmp_path / "comparison.png"
    
    path = plot_comparison(results, output_path=str(output_file))

    assert Path(path).exists()
    assert path == str(output_file)