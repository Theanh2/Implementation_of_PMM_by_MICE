"""
Plotting module for MICE (Multiple Imputation by Chained Equations)

This module combines diagnostic plots for analyzing imputed datasets 
and utilities for visualizing missing data patterns.
"""

# Import diagnostic plotting functions
from .diagnostics import (
    stripplot,
    bwplot, 
    densityplot,
    densityplot_split,
    xyplot,
    plot_chain_stats
)

# Import utility functions
from .utils import (
    md_pattern_like,
    plot_missing_data_pattern
)

__all__ = [
    # Diagnostic plots
    'stripplot',
    'bwplot',
    'densityplot', 
    'densityplot_split',
    'xyplot',
    'plot_chain_stats',
    # Utilities
    'md_pattern_like',
    'plot_missing_data_pattern'
] 