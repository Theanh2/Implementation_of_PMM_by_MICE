Multiple Imputation by Chained Equations in Python: mice-py
============================================================

This repository contains a modular Python framework for multiple imputation, developed as part of a master's thesis at Ludwig Maximilian University of Munich for the Statistics Department. The focus lies on implementing and evaluating **Predictive Mean Matching (PMM)** and its recent extension, the **midastouch** algorithm.

Overview
--------

The framework provides:

- **Flexible imputation methods**: PMM, MIDAS, CART, Random Forest, and random sampling approaches
- **Comprehensive diagnostics**: Visualization tools for analyzing imputation quality and missing data patterns
- **Professional logging**: Configurable logging system for monitoring imputation processes
- **Research-ready**: Built for statistical research with full reproducibility support

Key Features
------------

- **Multiple Imputation by Chained Equations (MICE)**: Full implementation with customizable parameters
- **Predictive Mean Matching (PMM)**: Traditional and midastouch variants
- **Tree-based Methods**: CART and Random Forest imputation
- **Diagnostic Tools**: Comprehensive plotting and analysis utilities

Getting Started
---------------

Installation
~~~~~~~~~~~~

Clone this repository:

.. code-block:: bash

   git clone https://github.com/Theanh2/Implementation_of_PMM_by_MICE.git
   cd Implementation_of_PMM_by_MICE

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import imputation
   import pandas as pd
   
   # Load your data with missing values
   data = pd.read_csv('your_data.csv')
   
   # Configure logging (optional)
   imputation.configure_logging()
   
   # Create MICE object and perform imputation
   mice = imputation.MICE(data)
   result = mice.impute(n_imputations=5, maxit=10)
   
   # Access imputed datasets
   imputed_data = result.complete_data(0)  # First imputation
   all_imputations = result.complete_data('all')  # All imputations

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: mice-py package

   imputation/index
   plotting/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
