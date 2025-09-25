#!/usr/bin/env python3
"""
Example script demonstrating the improved logging configuration for the imputation package.

This script shows how to configure logging for your own projects that use the imputation package.
"""

import pandas as pd
import numpy as np

# Example 1: Silent by default (no logging)
print("=== Example 1: Silent by default ===")
import imputation
print("Notice: No logging configuration needed - package is silent by default")

# Create some sample data
np.random.seed(42)
data = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100),
    'x3': np.random.normal(0, 1, 100)
})

# Add some missing values
data.loc[data.sample(frac=0.1).index, 'x1'] = np.nan
data.loc[data.sample(frac=0.1).index, 'x2'] = np.nan

# Create MICE object and run imputation - completely silent
mice = imputation.MICE(data)
mice.impute(n_imputations=3, maxit=2)
print("✓ Imputation completed silently (no log messages)")

print("\n" + "="*50)

# Example 2: Enable basic logging
print("\n=== Example 2: Enable basic logging ===")
imputation.configure_logging()
print("Logging enabled - now you'll see messages:")

# Run the same imputation - notice the logging behavior
mice2 = imputation.MICE(data)
mice2.impute(n_imputations=2, maxit=2)

print("\n" + "="*50)

# Example 3: File-only logging (quiet console)
print("\n=== Example 3: File-only logging ===")

# Configure for file-only logging
imputation.configure_logging(
    level='INFO',
    console=False,           # No console output
    file_logging=True        # Only log to file
)

print("Running imputation in quiet mode (check log file for messages)...")
mice3 = imputation.MICE(data)
mice3.impute(n_imputations=2, maxit=2)

print("\n" + "="*50)

# Example 4: Disable all logging
print("\n=== Example 4: Completely disable logging ===")

imputation.disable_logging()
print("Running imputation with logging disabled...")
mice4 = imputation.MICE(data)
mice4.impute(n_imputations=2, maxit=2)

print("\n" + "="*50)

# Example 5: Using individual module loggers
print("\n=== Example 5: Using module-specific loggers ===")

# Re-enable logging
imputation.configure_logging(level='INFO')

# Get logger for specific modules
cart_logger = imputation.get_logger('imputation.cart')
rf_logger = imputation.get_logger('imputation.rf')

cart_logger.info("This is a message from the CART module")
rf_logger.info("This is a message from the RF module")

# Example 6: Integration with your own project logging
print("\n=== Example 6: Integration with project logging ===")

import logging

# Set up your own project logger
project_logger = logging.getLogger('my_project')
project_handler = logging.StreamHandler()
project_handler.setFormatter(logging.Formatter('MY_PROJECT - %(levelname)s - %(message)s'))
project_logger.addHandler(project_handler)
project_logger.setLevel(logging.INFO)

# Configure imputation package logging separately
imputation.configure_logging(
    level='INFO',
    console_level='ERROR',  # Only show errors from imputation package on console
    file_logging=True       # But log everything to file
)

project_logger.info("Starting analysis with imputation package")
mice5 = imputation.MICE(data)
project_logger.info("MICE object created successfully")
mice5.impute(n_imputations=2, maxit=2)
project_logger.info("Imputation completed")

print("\n" + "="*70)
print("Examples completed!")
print("Check the following directories for log files:")
print("- ./logs/ (default location)")
print("- ./custom_logs/ (from Example 2)")
print("\nFor more information, see the documentation for:")
print("- imputation.configure_logging()")
print("- imputation.setup_logging()")
print("- imputation.get_logger()")
print("- imputation.disable_logging()")
print("- imputation.reset_logging()")
