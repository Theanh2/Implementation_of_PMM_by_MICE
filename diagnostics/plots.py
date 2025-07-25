import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
# Also set random seed for pandas
pd.set_option('mode.chained_assignment', None)  # Suppress pandas warnings
pd.set_option('display.max_columns', None)

def stripplot(imputed_datasets, missing_pattern, columns=None, merge_imputations=False, 
             observed_color='blue', imputed_color='red'):
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
    merge_imputations : bool, default False
        If True, shows two columns: one with only observed values and another with observed and imputed values overlaid.
        If False, shows separate plots for each imputation.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    
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
        
        if merge_imputations:
            # Add observed-only data
            plot_data.append(pd.DataFrame({
                'value': observed.values,
                'type': ['Observed'] * len(observed),
                'imputation': ['Observed Only'] * len(observed)
            }))
            
            # Add observed data again for the second column
            plot_data.append(pd.DataFrame({
                'value': observed.values,
                'type': ['Observed'] * len(observed),
                'imputation': ['Observed + Imputed'] * len(observed)
            }))
            
            # Combine all imputed values
            all_imp_values = []
            for df in imputed_datasets:
                imp_values = df.loc[~observed_mask, col]
                all_imp_values.extend(imp_values.values)
            
            # Add all imputed values to the second column
            plot_data.append(pd.DataFrame({
                'value': all_imp_values,
                'type': ['Imputed'] * len(all_imp_values),
                'imputation': ['Observed + Imputed'] * len(all_imp_values)
            }))
        else:
            # Add observed data
            plot_data.append(pd.DataFrame({
                'value': observed.values,
                'type': ['Observed'] * len(observed),
                'imputation': ['Observed'] * len(observed)
            }))
            
            # Add each imputation
            for i, df in enumerate(imputed_datasets, 1):
                # Get imputed values for this imputation
                imp_values = df.loc[~observed_mask, col]
                
                # Add observed and imputed values
                imp_data = pd.DataFrame({
                    'value': np.concatenate([observed.values, imp_values.values]),  # Use same observed values
                    'type': ['Observed'] * len(observed) + ['Imputed'] * len(imp_values),
                    'imputation': [f'Imp {i}'] * (len(observed) + len(imp_values))
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
            palette={'Observed': observed_color, 'Imputed': imputed_color},
            legend=False,
            alpha=0.6,
            linewidth=0.5
        )
        
        # Customize plot
        ax.set_title(f'{col}')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def bwplot(imputed_datasets, missing_pattern, columns=None, merge_imputations=False,
          observed_color='blue', imputed_color='red'):
    """
    Create box-and-whisker plots for imputed data showing observed and imputed values.
    First plots observed data, then for each imputation shows only imputed values
    in different colors.
    
    Parameters
    ----------
    imputed_datasets : list of pandas.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pandas.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    columns : list of str, optional
        List of column names to plot. If None, plots all columns with missing values.
    merge_imputations : bool, default False
        If True, combines all imputed values into a single boxplot. If False, shows separate boxplots for each imputation.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    
    Returns
    -------
    None
        Creates a figure with box-and-whisker plots
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
        
        # Create data for boxplot
        plot_data = []
        
        if merge_imputations:
            # Add observed data
            plot_data.append(pd.DataFrame({
                'value': observed.values,
                'type': ['Observed'] * len(observed),
                'imputation': ['Observed'] * len(observed)
            }))
            
            # Combine all imputed values
            all_imp_values = []
            for df in imputed_datasets:
                imp_values = df.loc[~observed_mask, col]
                all_imp_values.extend(imp_values.values)
            
            # Add all imputed values together
            plot_data.append(pd.DataFrame({
                'value': all_imp_values,
                'type': ['Imputed'] * len(all_imp_values),
                'imputation': ['Imputed'] * len(all_imp_values)
            }))
        else:
            # Add observed data
            plot_data.append(pd.DataFrame({
                'value': observed.values,
                'type': ['Observed'] * len(observed),
                'imputation': ['Observed'] * len(observed)
            }))
            
            # Add each imputation
            for i, df in enumerate(imputed_datasets, 1):
                # Get only imputed values for this imputation
                imp_values = df.loc[~observed_mask, col]
                
                # Add only imputed values
                imp_data = pd.DataFrame({
                    'value': imp_values.values,
                    'type': ['Imputed'] * len(imp_values),
                    'imputation': [f'Imp {i}'] * len(imp_values)
                })
                plot_data.append(imp_data)
        
        # Combine all data
        plot_data = pd.concat(plot_data, ignore_index=True)
        
        # Create boxplot
        sns.boxplot(
            data=plot_data,
            x='imputation',
            y='value',
            hue='type',
            ax=ax,
            palette={'Observed': observed_color, 'Imputed': imputed_color},
            legend=False,
            width=0.8,
            fill=False,
            showfliers=False,  # Hide outliers to avoid cluttering
            showbox=True,
            showcaps=True,
            showmeans=False,
            medianprops={'visible': False},  # Hide the median line
            boxprops={'alpha': 0.6},
            whiskerprops={'alpha': 0.6},
            capprops={'alpha': 0.6}
        )
        
        # Set transparency for all boxplot elements
        for patch in ax.artists:
            patch.set_alpha(0.6)
        
        # Make only whiskers dashed
        for line in ax.lines:
            # The whisker lines are the ones that extend beyond the box
            if len(line.get_xdata()) == 2:  # Whisker lines have 2 points
                line.set_linestyle('--')
                line.set_alpha(0.6)
        
        # Add median points
        if merge_imputations:
            # Add observed data point
            observed_data = plot_data[plot_data['imputation'] == 'Observed']
            if not observed_data.empty:
                median_val = observed_data['value'].median()
                ax.plot(0, median_val, 'o', color=observed_color, alpha=0.6, markersize=6, zorder=3)
            
            # Add imputed data point
            imp_data = plot_data[plot_data['imputation'] == 'Imputed']
            if not imp_data.empty:
                median_val = imp_data['value'].median()
                ax.plot(1, median_val, 'o', color=imputed_color, alpha=0.6, markersize=6, zorder=3)
        else:
            # Add observed data point
            observed_data = plot_data[plot_data['imputation'] == 'Observed']
            if not observed_data.empty:
                median_val = observed_data['value'].median()
                ax.plot(0, median_val, 'o', color=observed_color, alpha=0.6, markersize=6, zorder=3)
            
            # Then add imputed data points
            for i, df in enumerate(imputed_datasets, 1):
                imp_data = plot_data[(plot_data['imputation'] == f'Imp {i}') & (plot_data['type'] == 'Imputed')]
                if not imp_data.empty:
                    median_val = imp_data['value'].median()
                    ax.plot(i, median_val, 'o', color=imputed_color, alpha=0.6, markersize=6, zorder=3)
        
        # Customize plot
        ax.set_title(f'{col}')
        ax.set_ylabel('Value')
        ax.set_xlabel('')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def densityplot(imputed_datasets, missing_pattern, columns=None, 
               observed_color='blue', imputed_color='red'):
    """
    Create density plots (KDE) for observed and imputed data.
    Shows the distribution of observed data in blue and imputed data in red.
    
    Parameters
    ----------
    imputed_datasets : list of pandas.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pandas.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    columns : list of str, optional
        List of column names to plot. If None, plots all columns with missing values.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    
    Returns
    -------
    None
        Creates a figure with density plots
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
        
        # Plot observed data KDE
        sns.kdeplot(data=observed, ax=ax, color=observed_color, label='Observed', alpha=0.6, linewidth=2.5)
        
        # Plot imputed data KDE for each imputation
        for i, df in enumerate(imputed_datasets, 1):
            # Get only imputed values
            imp_values = df.loc[~observed_mask, col]
            # Only add label for the first imputation
            label = 'Imputed' if i == 1 else None
            sns.kdeplot(data=imp_values, ax=ax, color=imputed_color, label=label, alpha=0.6)
        
        # Customize plot
        ax.set_title(f'{col}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        
        # Handle legend - only for the first plot
        if col == columns[0]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:  # If we have any handles
                ax.legend(handles, labels, title='')
    
    plt.tight_layout()
    plt.show()

def densityplot_split(imputed_datasets, missing_pattern, column,
                     observed_color='blue', imputed_color='red'):
    """
    Create separate density plots (KDE) for observed data and each imputed dataset.
    Shows the distribution of observed data in blue and imputed data in red,
    with each imputation in a separate subplot.
    
    Parameters
    ----------
    imputed_datasets : list of pandas.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pandas.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    column : str
        Name of the column to plot
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    
    Returns
    -------
    None
        Creates a figure with density plots
    """
    if column not in missing_pattern.columns:
        print(f"Column {column} not found in the data")
        return
    
    if not missing_pattern[column].eq(0).any():
        print(f"No missing values in column {column}")
        return
    
    # Get observed values (where missing_pattern is 1)
    observed_mask = missing_pattern[column] == 1
    observed = imputed_datasets[0].loc[observed_mask, column]
    
    # Calculate number of plots and determine grid layout
    n_plots = len(imputed_datasets) + 1  # +1 for observed data
    
    # Determine number of rows and columns
    if n_plots <= 3:
        n_rows, n_cols = 1, n_plots
    elif n_plots <= 6:
        n_rows, n_cols = 2, (n_plots + 1) // 2
    elif n_plots <= 9:
        n_rows, n_cols = 3, (n_plots + 2) // 3
    else:
        n_rows, n_cols = 4, (n_plots + 3) // 4
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    
    # Plot observed data
    sns.kdeplot(data=observed, ax=axes[0], color=observed_color, alpha=0.6, linewidth=2.5)
    axes[0].set_title('Observed')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    
    # Plot each imputation
    for i, df in enumerate(imputed_datasets, 1):
        # Get only imputed values
        imp_values = df.loc[~observed_mask, column]
        
        # Check for zero variance
        if imp_values.nunique() <= 1:
            # If all values are the same, plot a vertical line
            value = imp_values.iloc[0]
            axes[i].axvline(x=value, color=imputed_color, alpha=0.6, linestyle='--')
            axes[i].set_title(f'Imp {i} (constant value: {value:.2f})')
        else:
            # If there's variance, plot the KDE
            sns.kdeplot(data=imp_values, ax=axes[i], color=imputed_color, alpha=0.6)
            axes[i].set_title(f'Imp {i}')
        
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')
    
    # Hide any unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Density plots for {column}', y=1.02)
    
    plt.tight_layout()
    plt.show()

def xyplot(imputed_datasets, missing_pattern, x, y, merge_imputations=False,
          observed_color='blue', imputed_color='red'):
    """
    Create scatter plots of two columns, showing observed and imputed values.
    Missing data in y is shown in red, observed data in blue.
    
    Parameters
    ----------
    imputed_datasets : list of pandas.DataFrame
        List of DataFrames containing imputed values
    missing_pattern : pandas.DataFrame
        DataFrame indicating missing values (0 where missing, 1 where observed)
    x : str
        Name of the column to plot on x-axis
    y : str
        Name of the column to plot on y-axis
    merge_imputations : bool, default False
        If True, shows all imputations on a single plot. If False, shows n+1 plots:
        first plot with only observed data, followed by one plot for each imputation.
    observed_color : str, default 'blue'
        Color for observed values
    imputed_color : str, default 'red'
        Color for imputed values
    
    Returns
    -------
    None
        Creates a figure with scatter plots
    """
    # Check if columns exist and provide specific error messages
    missing_cols = []
    if x not in missing_pattern.columns:
        missing_cols.append(f"x-axis column '{x}'")
    if y not in missing_pattern.columns:
        missing_cols.append(f"y-axis column '{y}'")
    
    if missing_cols:
        print(f"Error: The following columns are not found in the data: {', '.join(missing_cols)}")
        print(f"Available columns are: {', '.join(missing_pattern.columns)}")
        return
    
    # Get observed values (where missing_pattern is 1)
    observed_mask = missing_pattern[y] == 1
    
    if merge_imputations:
        # Create a single plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot observed data
        observed_x = imputed_datasets[0].loc[observed_mask, x]
        observed_y = imputed_datasets[0].loc[observed_mask, y]
        ax.scatter(observed_x, observed_y, color=observed_color, alpha=0.6, label=f'Observed')
        
        # Plot imputed data from all imputations
        for i, df in enumerate(imputed_datasets):
            imp_x = df.loc[~observed_mask, x]
            imp_y = df.loc[~observed_mask, y]
            # Only add label for the first imputation
            label = f'Imputed ({y})' if i == 0 else None
            ax.scatter(imp_x, imp_y, color=imputed_color, alpha=0.6, label=label)
        
        # Customize plot
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend()
        ax.set_title(f'Scatter plot of {x} vs {y} (all imputations)')
        
    else:
        # Calculate number of plots and determine grid layout
        n_plots = len(imputed_datasets) + 1  # +1 for observed data
        
        # Determine number of rows and columns
        if n_plots <= 3:
            n_rows, n_cols = 1, n_plots
        elif n_plots <= 6:
            n_rows, n_cols = 2, (n_plots + 1) // 2
        elif n_plots <= 9:
            n_rows, n_cols = 3, (n_plots + 2) // 3
        else:
            n_rows, n_cols = 4, (n_plots + 3) // 4
        
        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.flatten()  # Flatten the axes array for easier indexing
        
        # Plot observed data only in the first subplot
        observed_x = imputed_datasets[0].loc[observed_mask, x]
        observed_y = imputed_datasets[0].loc[observed_mask, y]
        axes[0].scatter(observed_x, observed_y, color=observed_color, alpha=0.6, label=f'Observed')
        axes[0].set_xlabel(x)
        axes[0].set_ylabel(y)
        axes[0].legend()
        axes[0].set_title('Observed')
        
        # Plot each imputation
        for i, df in enumerate(imputed_datasets, 1):
            # Plot observed data
            observed_x = df.loc[observed_mask, x]
            observed_y = df.loc[observed_mask, y]
            axes[i].scatter(observed_x, observed_y, color=observed_color, alpha=0.6, label=f'Observed ({y})')
            
            # Plot imputed data
            imp_x = df.loc[~observed_mask, x]
            imp_y = df.loc[~observed_mask, y]
            axes[i].scatter(imp_x, imp_y, color=imputed_color, alpha=0.6, label=f'Imputed ({y})')
            
            # Customize plot
            axes[i].set_xlabel(x)
            axes[i].set_ylabel(y)
            axes[i].legend()
            axes[i].set_title(f'Imp {i}')
        
        # Hide any unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Scatter plots of {x} vs {y}', y=1.02)
    
    plt.tight_layout()
    plt.show()