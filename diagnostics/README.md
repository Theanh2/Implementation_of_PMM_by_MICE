# Diagnostics

This folder contains diagnostics plots.

## Contents

- `plots.py`: Contains visualization functions for analyzing imputed datasets. For now just one is implemented.
  - `stripplot()`

- `example.ipynb`: Jupyter notebook demonstrating how to use the visualization tools with example data

## Usage

```python
from diagnostics.plots import stripplot

stripplot(
    imputed_datasets=[df1, df2, df3],  # List of imputed datasets as pandas df
    missing_pattern=missing_df,        # DataFrame indicating missing values (0=missing, 1=observed)
    columns=['col1', 'col2']           # Optional: specific columns to plot
)
```