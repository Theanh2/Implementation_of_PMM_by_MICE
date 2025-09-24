import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# True parameter values from the DGP: Y = 1 + 0.6·X1 - 0.3·X2 + 0.7·Z + 0.5·X1·Z + η
TRUE_PARAMETERS = {
    'Intercept': 1.0,
    'X1': 0.6,
    'X2': -0.3,
    'X3': 0.0,  # X3 is not in the DGP, so true value is 0
    'Z': 0.7,
    'X1:Z': 0.5
}

def load_pooled_results(folder_path):
    """Load pooled results from a folder"""
    results_file = os.path.join(folder_path, 'pooled_results.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            return json.load(f)
    return None

def calculate_statistics_for_folder(folder_path):
    """Calculate statistics for one folder (120 repetitions)"""
    print(f"Processing folder: {os.path.basename(folder_path)}")
    
    # Get all repetition folders
    rep_folders = sorted(glob.glob(os.path.join(folder_path, 'rep_*')))
    
    if not rep_folders:
        print(f"  No repetition folders found in {folder_path}")
        return None
    
    # Initialize storage for each parameter
    parameters = ['Intercept', 'X1', 'X2', 'X3', 'Z', 'X1:Z']
    stats = {param: {
        'estimates': [],
        'standard_errors': [],
        'lower_ci': [],
        'upper_ci': []
    } for param in parameters}
    
    # Collect data from all repetitions
    valid_reps = 0
    for rep_folder in rep_folders:
        results = load_pooled_results(rep_folder)
        if results is None:
            continue
            
        valid_reps += 1
        
        # Extract parameter estimates and CIs
        pooled_params = results['pooled_result']['params']
        pooled_cov = np.array(results['pooled_result']['cov_params'])
        
        # Calculate confidence intervals (95% CI)
        for i, param in enumerate(parameters):
            if i < len(pooled_params):
                estimate = pooled_params[i]
                se = np.sqrt(pooled_cov[i, i])
                
                stats[param]['estimates'].append(estimate)
                stats[param]['standard_errors'].append(se)
                stats[param]['lower_ci'].append(estimate - 1.96 * se)
                stats[param]['upper_ci'].append(estimate + 1.96 * se)
    
    print(f"  Processed {valid_reps} valid repetitions")
    
    # Calculate statistics for each parameter
    results = {}
    for param in parameters:
        if not stats[param]['estimates']:
            continue
            
        estimates = np.array(stats[param]['estimates'])
        lower_ci = np.array(stats[param]['lower_ci'])
        upper_ci = np.array(stats[param]['upper_ci'])
        true_value = TRUE_PARAMETERS[param]
        
        # 1. Raw Bias (RB)
        raw_bias = np.mean(estimates) - true_value
        
        # 2. Percent Bias (PB)
        if true_value != 0:
            percent_bias = (raw_bias / abs(true_value)) * 100
        else:
            percent_bias = np.nan
        
        # 3. Coverage Rate (CR)
        coverage = np.mean((lower_ci <= true_value) & (true_value <= upper_ci))
        
        # 4. Average Width (AW)
        avg_width = np.mean(upper_ci - lower_ci)
        
        # 5. Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean((estimates - true_value) ** 2))
        
        results[param] = {
            'raw_bias': raw_bias,
            'percent_bias': percent_bias,
            'coverage_rate': coverage,
            'avg_width': avg_width,
            'rmse': rmse,
            'n_repetitions': len(estimates)
        }
    
    # Save individual summary for this folder
    save_folder_summary(folder_path, results, valid_reps)
    
    return results

def save_folder_summary(folder_path, results, n_repetitions):
    """Save summary statistics for a single folder"""
    
    # Create summary data
    summary_data = []
    for param, stats in results.items():
        row = {
            'Parameter': param,
            'True_Value': TRUE_PARAMETERS[param],
            'Raw_Bias': stats['raw_bias'],
            'Percent_Bias': stats['percent_bias'],
            'Coverage_Rate': stats['coverage_rate'],
            'Avg_Width': stats['avg_width'],
            'RMSE': stats['rmse'],
            'N_Repetitions': stats['n_repetitions']
        }
        summary_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(summary_data)
    
    # Save as CSV
    csv_file = os.path.join(folder_path, "statistics_summary.csv")
    df.to_csv(csv_file, index=False)
    
    # Save as JSON
    json_file = os.path.join(folder_path, "statistics_summary.json")
    with open(json_file, 'w') as f:
        json.dump({
            'method': os.path.basename(folder_path),
            'n_repetitions': n_repetitions,
            'parameters': results,
            'summary_table': summary_data
        }, f, indent=2)
    
    print(f"  ✓ Saved summary to {csv_file}")
    print(f"  ✓ Saved summary to {json_file}")

def main():
    """Main function to process all folders"""
    base_path = "Simulate/Data/Imputations/MCAR_20"
    
    # Get all method folders
    method_folders = [f for f in os.listdir(base_path) 
                     if os.path.isdir(os.path.join(base_path, f)) and not f.startswith('.')]
    
    all_results = {}
    
    for method_folder in sorted(method_folders):
        folder_path = os.path.join(base_path, method_folder)
        print(f"\n{'='*80}")
        print(f"Processing method: {method_folder}")
        print(f"{'='*80}")
        
        results = calculate_statistics_for_folder(folder_path)
        if results:
            all_results[method_folder] = results
    
    # Save comprehensive results
    output_file = os.path.join(base_path, "statistics_summary.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Create summary table
    create_summary_table(all_results, base_path)

def create_summary_table(all_results, base_path):
    """Create a summary table of all statistics"""
    
    # Prepare data for table
    table_data = []
    
    for method, param_results in all_results.items():
        for param, stats in param_results.items():
            row = {
                'Method': method,
                'Parameter': param,
                'True_Value': TRUE_PARAMETERS[param],
                'Raw_Bias': stats['raw_bias'],
                'Percent_Bias': stats['percent_bias'],
                'Coverage_Rate': stats['coverage_rate'],
                'Avg_Width': stats['avg_width'],
                'RMSE': stats['rmse'],
                'N_Repetitions': stats['n_repetitions']
            }
            table_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    csv_file = os.path.join(base_path, "statistics_summary.csv")
    df.to_csv(csv_file, index=False)
    
    # Create summary by method and save to CSV
    method_summary = df.groupby('Method').agg({
        'Raw_Bias': ['mean', 'std'],
        'Percent_Bias': ['mean', 'std'],
        'Coverage_Rate': ['mean', 'std'],
        'Avg_Width': ['mean', 'std'],
        'RMSE': ['mean', 'std']
    }).round(4)
    method_summary.to_csv(os.path.join(base_path, "method_summary.csv"))
    
    # Create summary by parameter and save to CSV
    param_summary = df.groupby('Parameter').agg({
        'Raw_Bias': ['mean', 'std'],
        'Percent_Bias': ['mean', 'std'],
        'Coverage_Rate': ['mean', 'std'],
        'Avg_Width': ['mean', 'std'],
        'RMSE': ['mean', 'std']
    }).round(4)
    param_summary.to_csv(os.path.join(base_path, "parameter_summary.csv"))
    
    print(f"Summary tables saved to:")
    print(f"  Main CSV: {csv_file}")
    print(f"  Method summary: {os.path.join(base_path, 'method_summary.csv')}")
    print(f"  Parameter summary: {os.path.join(base_path, 'parameter_summary.csv')}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    print(f"\nOverall Coverage Rates by Method:")
    coverage_by_method = df.groupby('Method')['Coverage_Rate'].mean().sort_values(ascending=False)
    for method, coverage in coverage_by_method.items():
        print(f"  {method}: {coverage:.3f}")
    
    print(f"\nOverall RMSE by Method:")
    rmse_by_method = df.groupby('Method')['RMSE'].mean().sort_values()
    for method, rmse in rmse_by_method.items():
        print(f"  {method}: {rmse:.4f}")
    
    print(f"\nParameter-specific Coverage Rates:")
    coverage_by_param = df.groupby('Parameter')['Coverage_Rate'].mean().sort_values(ascending=False)
    for param, coverage in coverage_by_param.items():
        print(f"  {param}: {coverage:.3f}")

if __name__ == "__main__":
    main()
