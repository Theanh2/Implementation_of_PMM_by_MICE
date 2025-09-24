import os
import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
import sys

# Add the project root to the path to import MICE
sys.path.append('/Users/zhanna/Implementation_of_PMM_by_MICE')

from imputation.zh.MICE import MICE

def load_original_data():
    """Load the original simulated data"""
    data_path = "Simulate/Data/Inputs/initial_simulated_data.csv"
    return pd.read_csv(data_path)

def create_missing_data(original_data, mask):
    """Create missing data by applying mask to original data"""
    # Convert mask to boolean (1 = missing, 0 = observed)
    mask_bool = mask.astype(bool)
    
    # Create copy of original data
    missing_data = original_data.copy()
    
    # Apply mask: where mask is True, set to NaN
    missing_data[mask_bool] = np.nan
    
    return missing_data

def load_imputed_datasets(folder_path):
    """Load all imputed datasets from a folder"""
    # Find all imp_*.csv files
    imp_files = sorted(glob.glob(os.path.join(folder_path, "imp_*.csv")))
    
    if not imp_files:
        raise ValueError(f"No imputed datasets found in {folder_path}")
    
    # Load each dataset
    datasets = []
    for file_path in imp_files:
        dataset = pd.read_csv(file_path)
        datasets.append(dataset)
    
    return datasets

def calculate_performance_metrics(pooled_result, original_data, missing_data):
    """Calculate performance metrics for the pooled results"""
    
    # True parameters from the DGP: Y = 1 + 0.6·X1 - 0.3·X2 + 0.7·Z + 0.5·X1·Z + η
    true_params = {
        'Intercept': 1.0,
        'X1': 0.6,
        'X2': -0.3,
        'X3': 0.0,  # X3 is not in the true DGP
        'Z': 0.7,
        'X1:Z': 0.5  # Interaction term
    }
    
    # Get pooled estimates
    pooled_params = pooled_result.params  # This is a numpy array
    pooled_cov = pooled_result.cov_params()  # This is a numpy array
    
    # Get parameter names from the model
    param_names = list(pooled_result.params.index) if hasattr(pooled_result.params, 'index') else [
        'Intercept', 'X1', 'X2', 'X3', 'Z', 'X1:Z'
    ]
    
    # Calculate metrics for each parameter
    parameter_metrics = {}
    overall_metrics = {
        'bias_mean': 0.0,
        'width_mean': 0.0,
        'coverage_rate': 0.0,
        'mse_mean': 0.0
    }
    
    valid_params = 0
    
    for i, param_name in enumerate(param_names):
        if param_name in true_params:
            estimated_param = pooled_params[i]
            estimated_se = np.sqrt(pooled_cov[i, i])
            true_param = true_params[param_name]
            
            # Calculate metrics for this parameter
            bias = estimated_param - true_param
            width = 2 * 1.96 * estimated_se  # 95% CI width
            coverage = abs(bias) <= 1.96 * estimated_se  # True if 0 is in 95% CI
            
            parameter_metrics[param_name] = {
                'bias': bias,
                'width': width,
                'coverage': coverage,
                'estimated_param': estimated_param,
                'estimated_se': estimated_se,
                'true_param': true_param
            }
            
            # Accumulate for overall metrics
            overall_metrics['bias_mean'] += abs(bias)
            overall_metrics['width_mean'] += width
            overall_metrics['coverage_rate'] += coverage
            overall_metrics['mse_mean'] += bias**2
            valid_params += 1
    
    # Average the overall metrics
    if valid_params > 0:
        overall_metrics['bias_mean'] /= valid_params
        overall_metrics['width_mean'] /= valid_params
        overall_metrics['coverage_rate'] /= valid_params
        overall_metrics['mse_mean'] /= valid_params
    
    # Calculate actual missing proportion
    actual_missing = missing_data['Y'].isna().mean()
    
    return {
        'parameter_metrics': parameter_metrics,
        'overall_metrics': overall_metrics,
        'actual_missing': actual_missing,
        'frac_miss_info': pooled_result.frac_miss_info.tolist() if hasattr(pooled_result, 'frac_miss_info') else None
    }

def analyze_folder(folder_path, original_data):
    """Analyze a single folder"""
    print(f"Analyzing folder: {folder_path}")
    
    try:
        # Load mask
        mask_path = os.path.join(folder_path, "mask.csv")
        mask = pd.read_csv(mask_path)
        
        # Create missing data
        missing_data = create_missing_data(original_data, mask)
        
        # Load imputed datasets
        imputed_datasets = load_imputed_datasets(folder_path)
        
        # Create MICE object
        mice_obj = MICE(missing_data)
        
        # Assign imputed datasets
        mice_obj.imputed_datasets = imputed_datasets
        
        # Fit model (Y ~ X1 + X2 + X3 + Z + X1:Z)
        mice_obj.fit('Y ~ X1 + X2 + X3 + Z + X1:Z')
        
        # Pool results
        comprehensive_result = mice_obj.pool()
        pooled_result = comprehensive_result['pooled_result']
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(pooled_result, original_data, missing_data)
        
        # Save results
        results = {
            'pooled_result': {
                'params': pooled_result.params.tolist(),
                'cov_params': pooled_result.cov_params().tolist(),
                'scale': pooled_result.scale,
                'frac_miss_info': pooled_result.frac_miss_info.tolist() if hasattr(pooled_result, 'frac_miss_info') else None
            },
            'comprehensive_results': {
                'pooled_params': comprehensive_result['pooled_params'].tolist(),
                'pooled_covariance': comprehensive_result['pooled_covariance'].tolist(),
                'within_covariance': comprehensive_result['within_covariance'].tolist(),
                'between_covariance': comprehensive_result['between_covariance'].tolist(),
                'fraction_missing_info': comprehensive_result['fraction_missing_info'].tolist(),
                'pooled_scale': comprehensive_result['pooled_scale'],
                'n_imputations': comprehensive_result['n_imputations'],
                'parameter_names': comprehensive_result['parameter_names'],
                'formula': comprehensive_result['formula']
            },
            'performance_metrics': metrics,
            'metadata': {
                'n_imputations': len(imputed_datasets),
                'n_observations': len(missing_data),
                'formula': 'Y ~ X1 + X2 + X3 + Z + X1:Z'
            }
        }
        
        # Save to file
        results_path = os.path.join(folder_path, "pooled_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"  ✓ Results saved to {results_path}")
        print(f"  ✓ Overall Bias (mean): {metrics['overall_metrics']['bias_mean']:.6f}")
        print(f"  ✓ Overall Width (mean): {metrics['overall_metrics']['width_mean']:.6f}")
        print(f"  ✓ Coverage Rate: {metrics['overall_metrics']['coverage_rate']:.3f}")
        print(f"  ✓ Overall MSE (mean): {metrics['overall_metrics']['mse_mean']:.6f}")
        print(f"  ✓ Actual Missing: {metrics['actual_missing']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error analyzing {folder_path}: {str(e)}")
        return None

def main():
    """Main function to analyze all folders"""
    
    # Load original data
    print("Loading original data...")
    original_data = load_original_data()
    print(f"Original data shape: {original_data.shape}")
    
    # Base directory
    base_dir = "Simulate/Data/Imputations/MCAR_20"
    
    # Get all method folders
    method_folders = [f for f in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    
    print(f"Found {len(method_folders)} method folders")
    
    all_results = {}
    
    for method_folder in method_folders:
        method_path = os.path.join(base_dir, method_folder)
        print(f"\nProcessing method: {method_folder}")
        
        # Get all replicate folders
        replicate_folders = [f for f in os.listdir(method_path) 
                           if os.path.isdir(os.path.join(method_path, f)) and f.startswith('rep_')]
        
        print(f"  Found {len(replicate_folders)} replicates")
        
        method_results = []
        
        for replicate_folder in replicate_folders:
            replicate_path = os.path.join(method_path, replicate_folder)
            
            # Check if results already exist
            results_path = os.path.join(replicate_path, "pooled_results.json")
            if os.path.exists(results_path):
                print(f"  ✓ Skipping {replicate_folder} (results already exist)")
                continue
            
            # Analyze this replicate
            result = analyze_folder(replicate_path, original_data)
            if result:
                method_results.append(result)
        
        all_results[method_folder] = method_results
    
    # Save summary
    summary_path = os.path.join(base_dir, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nAnalysis complete! Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
