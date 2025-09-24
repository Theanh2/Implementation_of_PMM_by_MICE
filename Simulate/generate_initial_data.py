"""
Script to generate initial simulated dataset with specified mechanism.

Data generation mechanism:
- X1 ~ N(0,1)
- X2 ~ N(0,1)  
- X3 = 0.5·X1² + 0.7·X2 + ε, ε~N(0,1) (nonlinear)
- Z (binary) with logit P(Z=1) = −0.4 + 0.8·X1 − 0.6·X2
- Y (continuous target estimand) = 1 + 0.6·X1 − 0.3·X2 + 0.7·Z + 0.5·X1·Z + η, η~N(0,1)

Generates 1000 rows with reproducible random state.
"""

import numpy as np
import pandas as pd
from scipy.special import expit  
import os

def generate_simulated_data(n_samples=1000, random_state=42):
    """
    Generate simulated dataset according to specified mechanism.
    
    Parameters:
    -----------
    n_samples : int, default=1000
        Number of samples to generate
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Generated dataset with columns X1, X2, X3, Z, Y
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Generate X1 and X2 from standard normal distribution
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    
    # Generate X3 (nonlinear relationship)
    epsilon = np.random.normal(0, 1, n_samples)
    X3 = 0.5 * X1**2 + 0.7 * X2 + epsilon
    
    # Generate Z (binary) using logistic regression
    # logit P(Z=1) = -0.4 + 0.8·X1 - 0.6·X2
    logit_z = -0.4 + 0.8 * X1 - 0.6 * X2
    prob_z = expit(logit_z)  # Convert logit to probability using logistic function
    Z = np.random.binomial(1, prob_z, n_samples)
    
    # Generate Y (continuous target estimand)
    # Y = 1 + 0.6·X1 - 0.3·X2 + 0.7·Z + 0.5·X1·Z + η
    eta = np.random.normal(0, 1, n_samples)
    Y = 1 + 0.6 * X1 - 0.3 * X2 + 0.7 * Z + 0.5 * X1 * Z + eta
    
    # Create DataFrame
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'X3': X3,
        'Z': Z,
        'Y': Y
    })
    
    return data

def main():
    """Main function to generate and save the dataset."""
    print("Generating simulated dataset...")
    
    # Generate the data
    data = generate_simulated_data(n_samples=1000, random_state=42)
    
    # Print summary statistics
    print(f"Dataset shape: {data.shape}")
    print("\nSummary statistics:")
    print(data.describe())
    print(f"\nZ distribution (binary): {data['Z'].value_counts().sort_index()}")
    
    # Save to CSV in the Inputs directory
    output_path = "Data/Inputs/initial_simulated_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to: {output_path}")
    
    # Verify the file was created
    if os.path.exists(output_path):
        print("✓ File successfully created")
    else:
        print("✗ Error: File was not created")

if __name__ == "__main__":
    main()
