import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def stripplot(imputed_datasets, missing_pattern, columns=None):
    """
    Create stripplots for imputed data showing observed and imputed values.
    First plots observed data, then for each imputation shows both observed and imputed values
    in different colors.
    
    Parameters
    ----------
    imputed_datasets : list of pandas.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pandas.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    columns : list of str, optional
        List of column names to plot. If None, plots all columns with missing values.
    
    Returns
    -------
    None
        Creates a figure with stripplots
    """
    # If no columns specified, use all columns with missing values
    if columns is None:
        columns = missing_pattern.columns[missing_pattern.eq(0).any()]
    
    # Filter columns to only those with missing values
    columns = [col for col in columns if col in missing_pattern.columns and missing_pattern[col].eq(0).any()]
    
    if not columns:
        print("No columns with missing values to plot")
        return
    
    # Create a figure with subplots
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    # Plot each column
    for ax, col in zip(axes, columns):
        # Get observed values (where missing_pattern is 1)
        observed_mask = missing_pattern[col] == 1
        observed = imputed_datasets[0].loc[observed_mask, col]
        
        # Create data for stripplot
        plot_data = []
        
        # Add observed data
        plot_data.append(pd.DataFrame({
            'value': observed.values,
            'type': ['Observed'] * len(observed),
            'imputation': ['Observed'] * len(observed)
        }))
        
        # Add each imputation
        for i, df in enumerate(imputed_datasets, 1):
            # Get observed values for this imputation
            obs_values = df.loc[observed_mask, col]
            # Get imputed values for this imputation
            imp_values = df.loc[~observed_mask, col]
            
            # Combine observed and imputed values
            imp_data = pd.DataFrame({
                'value': np.concatenate([obs_values.values, imp_values.values]),
                'type': ['Observed'] * len(obs_values) + ['Imputed'] * len(imp_values),
                'imputation': [f'Imp {i}'] * (len(obs_values) + len(imp_values))
            })
            plot_data.append(imp_data)
        
        # Combine all data
        plot_data = pd.concat(plot_data, ignore_index=True)
        
        # Create stripplot
        sns.stripplot(
            data=plot_data,
            x='imputation',
            y='value',
            hue='type',
            ax=ax,
            jitter=True,
            palette={'Observed': 'blue', 'Imputed': 'red'},
            legend=False,
            alpha=0.6,
            linewidth=0.5
        )
        
        # Customize plot
        ax.set_title(f'{col}')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        ax.legend(title='')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()