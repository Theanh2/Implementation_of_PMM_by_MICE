"""
Showcase script for the MICE imputation framework.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from zh.MICE import MICE
from zh.constants import ImputationMethod, InitialMethod, SUPPORTED_METHODS, SUPPORTED_INITIAL_METHODS
from visualization.utils import md_pattern_like, plot_missing_data_pattern


def load_nhanes_data():
    """Load the NHANES dataset."""
    data_path = Path(__file__).parent.parent.parent / "data" / "nhanes.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"NHANES data file not found at {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"✅ Loaded NHANES dataset: {data.shape[0]} rows × {data.shape[1]} columns")
    
    return data


def create_missing_data_plot(data):
    """Create missing data pattern plot using utils.py functions."""
    print("\n📊 Creating missing data pattern plot...")
    
    pattern_df = md_pattern_like(data)
    
    plot_missing_data_pattern(pattern_df, figsize=(10, 6), title="NHANES Dataset Missing Data Pattern")
    
    print("✅ Missing data pattern plot created successfully!")


def print_dataset_summary(data):
    """Print a comprehensive summary of the dataset."""
    print("\n📋 Dataset Summary:")
    print("=" * 50)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Variables: {list(data.columns)}")
    
    print("\nData types:")
    for col in data.columns:
        print(f"  {col}: {data[col].dtype}")
    
    print("\nMissing values:")
    missing_counts = data.isnull().sum()
    missing_percent = (missing_counts / len(data)) * 100
    
    for col in data.columns:
        count = missing_counts[col]
        percent = missing_percent[col]
        print(f"  {col}: {count} ({percent:.1f}%)")
    
    print("\nAvailable imputation methods:")
    print(f"  {SUPPORTED_METHODS}")
    
    print("\nAvailable initial methods:")
    print(f"  {SUPPORTED_INITIAL_METHODS}")


def initialize_mice_object(data):
    """Initialize a MICE object with the data."""
    print("\n🔧 Initializing MICE object...")
    
    try:
        mice_obj = MICE(data)
        
        print("✅ MICE object initialized successfully!")
        print(f"  - Cleaned data shape: {mice_obj.data.shape}")
        print(f"  - Variables: {list(mice_obj.data.columns)}")
        
        if len(mice_obj.data.columns) < len(data.columns):
            dropped_cols = set(data.columns) - set(mice_obj.data.columns)
            print(f"  - Dropped columns: {list(dropped_cols)}")
        
        return mice_obj
        
    except Exception as e:
        print(f"❌ Error initializing MICE object: {str(e)}")
        return None


def main():
    """Main function to run the showcase."""
    print("🚀 MICE Imputation Framework Showcase")
    print("=" * 50)
    
    try:
        data = load_nhanes_data()
        
        print_dataset_summary(data)
        
        create_missing_data_plot(data)
        
        mice_obj = initialize_mice_object(data)
        
    except Exception as e:
        print(f"❌ Error in showcase: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 