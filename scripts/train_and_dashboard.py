#!/usr/bin/env python3
"""
Convenient script to train all models and automatically open MLflow dashboard.

This script:
1. Runs the complete model training pipeline
2. Automatically opens MLflow dashboard in your default browser
3. Provides real-time training progress updates
4. Shows model performance summary after completion

Usage:
    python scripts/train_and_dashboard.py [--no-browser] [--port 5000]

Options:
    --no-browser    Skip opening browser automatically
    --port          MLflow UI port (default: 5000)
    --host          MLflow UI host (default: localhost)
    --log-level     Logging level (default: INFO)
"""

import sys
import os
import time
import webbrowser
import subprocess
import threading
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging, get_logger


def run_training():
    """Run the training script and return the exit code."""
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting model training pipeline...")
    
    try:
        # Use the simple training script that works
        training_script = project_root / "scripts" / "train_simple.py"
        if training_script.exists():
            logger.info("Running scripts/train_simple.py...")
            result = subprocess.run([sys.executable, str(training_script)], cwd=project_root)
            return result.returncode
        
        # Fallback to the original script
        training_script = project_root / "scripts" / "train_models.py"
        if training_script.exists():
            logger.info("Running scripts/train_models.py...")
            result = subprocess.run([sys.executable, str(training_script)], cwd=project_root)
            return result.returncode
        
        # Last resort: try importing the training function directly
        logger.info("Trying to import training function directly...")
        sys.path.append(str(project_root / "src"))
        from models.train_all import main as train_main
        result = train_main()
        
        if result == 0:
            logger.info("✅ Model training completed successfully!")
        else:
            logger.error("❌ Model training failed!")
            
        return result
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        return 1


def start_mlflow_ui(host="localhost", port=5000):
    """Start MLflow UI in a separate thread."""
    logger = get_logger(__name__)
    
    def run_mlflow():
        try:
            # Change to project root directory
            os.chdir(project_root)
            
            # Run MLflow UI
            logger.info(f"🌐 Starting MLflow UI at http://{host}:{port}")
            
            # Use subprocess to run MLflow UI
            subprocess.run([
                sys.executable, "-m", "mlflow", "ui",
                "--host", host,
                "--port", str(port),
                "--backend-store-uri", "./mlruns"
            ], check=False)
            
        except Exception as e:
            logger.error(f"Failed to start MLflow UI: {e}")
    
    # Start MLflow UI in background thread
    thread = threading.Thread(target=run_mlflow, daemon=True)
    thread.start()
    
    return f"http://{host}:{port}"


def wait_for_mlflow_ui(url, max_wait=30):
    """Wait for MLflow UI to become available."""
    import requests
    logger = get_logger(__name__)
    
    logger.info(f"⏳ Waiting for MLflow UI to start at {url}...")
    
    for i in range(max_wait):
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                logger.info("✅ MLflow UI is ready!")
                return True
        except:
            pass
        
        time.sleep(1)
        if i % 5 == 0 and i > 0:
            logger.info(f"Still waiting for MLflow UI... ({i}/{max_wait}s)")
    
    logger.warning(f"⚠️  MLflow UI didn't start within {max_wait} seconds")
    return False


def open_browser(url):
    """Open the URL in the default browser."""
    logger = get_logger(__name__)
    
    try:
        logger.info(f"🌐 Opening MLflow dashboard in your browser: {url}")
        webbrowser.open(url)
        return True
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")
        logger.info(f"Please manually open: {url}")
        return False


def display_training_summary():
    """Display a summary of trained models from MLflow."""
    logger = get_logger(__name__)
    
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("./mlruns")
        client = MlflowClient()
        
        # Get experiment
        experiment = client.get_experiment_by_name("california_housing_prediction")
        if not experiment:
            logger.warning("No experiment found")
            return
        
        # Get latest runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10
        )
        
        if not runs:
            logger.warning("No training runs found")
            return
        
        logger.info("\n" + "="*70)
        logger.info("📊 TRAINING SUMMARY")
        logger.info("="*70)
        
        # Group runs by model type
        model_summary = {}
        for run in runs:
            if run.data.tags.get("model_type"):
                model_type = run.data.tags["model_type"]
                r2_score = run.data.metrics.get("r2_score", 0)
                
                if model_type not in model_summary or r2_score > model_summary[model_type]["r2_score"]:
                    model_summary[model_type] = {
                        "r2_score": r2_score,
                        "rmse": run.data.metrics.get("rmse", 0),
                        "run_id": run.info.run_id,
                        "start_time": run.info.start_time
                    }
        
        # Display summary
        for model_type, metrics in sorted(model_summary.items(), 
                                        key=lambda x: x[1]["r2_score"], reverse=True):
            logger.info(f"🏆 {model_type.upper():<15} - R² = {metrics['r2_score']:.3f}, RMSE = {metrics['rmse']:.3f}")
        
        logger.info("="*70)
        logger.info(f"✅ {len(model_summary)} models trained and registered in MLflow")
        
    except Exception as e:
        logger.error(f"Failed to display training summary: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train models and launch MLflow dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Skip opening browser automatically"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="MLflow UI port (default: 5000)"
    )
    
    parser.add_argument(
        "--host",
        default="localhost",
        help="MLflow UI host (default: localhost)"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only launch MLflow dashboard"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file="logs/training.log")
    logger = get_logger(__name__)
    
    logger.info("🎯 California Housing MLOps Training & Dashboard Launcher")
    logger.info("="*70)
    
    training_success = True
    
    # Step 1: Run training (unless skipped)
    if not args.skip_training:
        start_time = time.time()
        training_result = run_training()
        training_time = time.time() - start_time
        
        if training_result == 0:
            logger.info(f"✅ Training completed in {training_time:.1f} seconds")
            display_training_summary()
        else:
            logger.error("❌ Training failed!")
            training_success = False
    else:
        logger.info("⏭️  Skipping training as requested")
    
    # Step 2: Start MLflow UI
    logger.info("\n" + "="*70)
    logger.info("🚀 LAUNCHING MLFLOW DASHBOARD")
    logger.info("="*70)
    
    mlflow_url = start_mlflow_ui(args.host, args.port)
    
    # Step 3: Wait for MLflow UI to be ready
    if wait_for_mlflow_ui(mlflow_url):
        
        # Step 4: Open browser (unless disabled)
        if not args.no_browser:
            open_browser(mlflow_url)
        
        logger.info("\n" + "="*70)
        logger.info("🎉 SETUP COMPLETE!")
        logger.info("="*70)
        logger.info(f"🌐 MLflow Dashboard: {mlflow_url}")
        logger.info("📊 View experiments, models, and metrics")
        logger.info("🔧 Compare model performance and parameters")
        logger.info("📈 Track training progress and artifacts")
        logger.info("\n💡 Press Ctrl+C to stop the MLflow server")
        logger.info("="*70)
        
        try:
            # Keep the script running to maintain MLflow UI
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\n👋 Shutting down MLflow dashboard...")
            
    else:
        logger.error("❌ Failed to start MLflow UI")
        return 1
    
    return 0 if training_success else 1


if __name__ == "__main__":
    exit(main()) 