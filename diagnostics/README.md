# Diagnostics

This folder contains diagnostics plots.

## Contents

- `plots.py`: Contains visualization functions for analyzing imputed datasets:
  - `stripplot()`: Creates stripplots showing observed and imputed values
  - `bwplot()`: Creates box-and-whisker plots for observed and imputed values
  - `densityplot()`: Creates density plots (KDE) for observed and imputed values
  - `densityplot_split()`: Creates separate density plots for one column for observed data and each imputation
  - `xyplot()`: Creates scatter plots of two columns, showing observed and imputed values

  All functions support:
  - Custom colors for observed and imputed values
  - Option to merge imputations into a single plot or show separate plots
  - Automatic handling of missing values

- `example.ipynb`: Jupyter notebook demonstrating how to use the visualization tools with example data

## Usage example

```python
from diagnostics.plots import stripplot

stripplot(
    imputed_datasets=[df1, df2, df3],  # List of imputed datasets as pandas df
    missing_pattern=missing_df,        # DataFrame indicating missing values (0=missing, 1=observed)
    columns=['col1', 'col2']           # Optional: specific columns to plot
)
```
