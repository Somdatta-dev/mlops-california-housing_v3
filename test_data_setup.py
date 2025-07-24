#!/usr/bin/env python3
"""
Simple test script to download California Housing data.
"""
import pandas as pd
from sklearn.datasets import fetch_california_housing
from pathlib import Path
import os

def main():
    print("Starting data download...")
    
    # Create data directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download California Housing dataset
        print("Downloading California Housing dataset...")
        housing = fetch_california_housing(as_frame=True)
        
        # Get the full dataset
        df = housing.frame
        
        # Save raw data
        raw_file = raw_dir / "california_housing.csv"
        df.to_csv(raw_file, index=False)
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"   - Shape: {df.shape}")
        print(f"   - Saved to: {raw_file}")
        print(f"   - Size: {raw_file.stat().st_size / 1024:.1f} KB")
        
        # Show first few rows
        print("\n📊 Dataset preview:")
        print(df.head())
        
        # Show dataset info
        print(f"\n📈 Dataset statistics:")
        print(f"   - Rows: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Features: {list(df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 