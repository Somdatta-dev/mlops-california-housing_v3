#!/usr/bin/env python3
"""
Script to run the California Housing Price Prediction API.
"""

import os
import sys
import argparse
import uvicorn
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.api.config import get_config, setup_environment
from src.utils.logging_config import setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the California Housing Price Prediction API"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (default: from config)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (default: from config)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: from config)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Log level (default: from config)"
    )
    
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to environment file (default: .env)"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the API."""
    args = parse_args()
    
    # Setup environment
    if args.env_file and os.path.exists(args.env_file):
        os.environ["ENV_FILE"] = args.env_file
    
    config = setup_environment()
    
    # Override config with command line arguments
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.workers:
        config.workers = args.workers
    if args.reload:
        config.reload = True
    if args.debug:
        config.debug = True
    if args.log_level:
        config.log_level = args.log_level
    
    # Setup logging
    setup_logging(
        log_level=config.log_level,
        log_file=config.log_file
    )
    logger = get_logger(__name__)
    
    logger.info("Starting California Housing Price Prediction API")
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"Workers: {config.workers}")
    logger.info(f"Reload: {config.reload}")
    logger.info(f"Debug: {config.debug}")
    logger.info(f"Log Level: {config.log_level}")
    
    # Run the server
    try:
        uvicorn.run(
            "src.api.main:app",
            host=config.host,
            port=config.port,
            workers=config.workers if not config.reload else 1,
            reload=config.reload,
            log_level=config.log_level.lower(),
            access_log=True,
            server_header=False,
            date_header=False
        )
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 