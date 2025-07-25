import os
from typing import List, Tuple, Dict, Any

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.mlflow_config import MLflowConfig

class ModelComparator:
    """
    Automate comparison of multiple MLflow-registered models using cross-validation,
    statistical significance testing, and multi-metric selection logic.
    """

    def __init__(
        self,
        model_names: List[str],
        model_versions: List[int],
        mlflow_config: MLflowConfig = None,
        metrics: List[str] = None,
        cv_splits: int = 5,
        random_state: int = 42,
    ):
        if len(model_names) != len(model_versions):
            raise ValueError("model_names and model_versions must have the same length.")
        self.model_names = model_names
        self.model_versions = model_versions
        self.mlflow = mlflow_config or MLflowConfig()
        self.metrics = metrics or ["rmse", "mae", "r2"]
        self.cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def _load_model(self, name: str, version: int):
        uri = f"models:/{name}/{version}"
        return mlflow.pyfunc.load_model(uri)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """
        Perform cross-validated evaluation for each model.
        Returns a DataFrame with mean and std for each metric.
        """
        records = []
        for name, version in zip(self.model_names, self.model_versions):
            model = self._load_model(name, version)
            scores = {m: [] for m in self.metrics}

            # cross_validate returns values for scoring keys, but we use manual loop
            for train_idx, test_idx in self.cv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                preds = model.predict(X_te)

                if "rmse" in scores:
                    scores["rmse"].append(np.sqrt(mean_squared_error(y_te, preds)))
                if "mae" in scores:
                    scores["mae"].append(mean_absolute_error(y_te, preds))
                if "r2" in scores:
                    scores["r2"].append(r2_score(y_te, preds))

            # aggregate
            agg = {"model": f"{name}:{version}"}
            for m in self.metrics:
                arr = np.array(scores[m])
                agg[f"{m}_mean"] = arr.mean()
                agg[f"{m}_std"] = arr.std()
            records.append(agg)

        return pd.DataFrame.from_records(records)

    def statistical_tests(
        self, X: pd.DataFrame, y: pd.Series, results: pd.DataFrame, metric: str
    ) -> Dict[str, Any]:
        """
        Perform paired t-tests between the best model and all others for a given metric.
        Returns dict of p-values keyed by model.
        """
        # find best mean for metric
        best = results.loc[results[f"{metric}_mean"].idxmin() if metric != "r2" else results[f"{metric}_mean"].idxmax()]
        best_name = best["model"]
        p_values = {}
        # reload raw fold scores
        base_scores = []
        for name, version in zip(self.model_names, self.model_versions):
            model = self._load_model(name, version)
            folds = []
            for train_idx, test_idx in self.cv.split(X, y):
                preds = model.predict(X.iloc[test_idx])
                if metric == "rmse":
                    folds.append(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
                elif metric == "mae":
                    folds.append(mean_absolute_error(y.iloc[test_idx], preds))
                else:
                    folds.append(r2_score(y.iloc[test_idx], preds))
            if name + f":{version}" == best_name:
                base_scores = folds
        # compare
        for rec in results.to_dict('records'):
            other_name = rec["model"]
            if other_name == best_name:
                continue
            other_scores = []
            nm, ver = other_name.split(":")
            mdl = self._load_model(nm, int(ver))
            for train_idx, test_idx in self.cv.split(X, y):
                preds = mdl.predict(X.iloc[test_idx])
                if metric == "rmse":
                    other_scores.append(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
                elif metric == "mae":
                    other_scores.append(mean_absolute_error(y.iloc[test_idx], preds))
                else:
                    other_scores.append(r2_score(y.iloc[test_idx], preds))
            stat, p = ttest_rel(base_scores, other_scores)
            p_values[other_name] = p
        return {"best_model": best_name, "p_values": p_values}

    def select_best(
        self, results: pd.DataFrame, priority: List[str] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Select the best model based on a list of prioritized metrics.
        For metrics where lower is better (rmse, mae), chooses minimal mean;
        for r2, chooses maximal.
        """
        df = results.copy()
        priority = priority or self.metrics
        # lexicographic sort by priority
        sort_cols = []
        ascending = []
        for m in priority:
            sort_cols.append(f"{m}_mean")
            ascending.append(False if m == "r2" else True)
        best_record = df.sort_values(sort_cols, ascending=ascending).iloc[0]
        scores = {m: best_record[f"{m}_mean"] for m in self.metrics}
        return best_record["model"], scores

    def register_best(
        self, best_model: str, dest_name: str, description: str = "", stage: str = "Staging"
    ) -> None:
        """
        Register and stage the best-performing model in MLflow Model Registry.
        """
        uri = f"models:/{best_model.replace(':', '/')}"
        model_version = self.mlflow.register_model(model_uri=uri, model_name=dest_name, description=description)
        # promote immediately to desired stage
        self.mlflow.promote_model(
            model_name=dest_name, version=model_version.version, stage=stage
        )

def plot_comparison(
    results: pd.DataFrame,
    metrics: List[str] = None,
    output_path: str = "model_comparison.png"
) -> str:
    """
    Generate a bar chart comparing models on specified metrics.
    Saves to output_path and returns its path.
    """
    metrics = metrics or ["rmse_mean", "mae_mean", "r2_mean"]
    df = results.set_index("model")
    df_plot = df[metrics]
    ax = df_plot.plot.bar(rot=45, figsize=(8, 6), title="Model Comparison")
    ax.set_ylabel("Score")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    return output_path