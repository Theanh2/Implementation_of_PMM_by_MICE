.. Imputepmmidas documentation master file, created by
   sphinx-quickstart on Sun Jun 22 19:37:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Imputepmmidas documentation
===========================

Implementation of PMM and Variations
====================================

This repository contains a modular Python framework for multiple imputation, developed as part of my master's thesis at Ludwig Maximilian University of Munich for the Statistics Department. The focus lies on implementing and evaluating **Predictive Mean Matching (PMM)** and its recent extension, the **midastouch** algorithm.

Thesis Abstract
---------------

This thesis develops a modular Python framework for multiple imputation inspired by the R package ``mice``, with a particular focus on **Predictive Mean Matching (PMM)** and the **midastouch** algorithm. The implementation allows for full flexibility in defining:

- Distance metrics
- Donor selection rules
- Imputation parameters

To evaluate both imputation strategies, a comprehensive simulation study was conducted across three types of target variables:

- **Continuous**
- **Semi-continuous**
- **Discrete**

Each simulation varied by:

- **Missingness mechanism**: MCAR (Missing Completely At Random), left-tailed MAR, and right-tailed MAR
- **Proportion of missing data**
- **Imputation configuration**

In total, 675 configurations were simulated and evaluated using four key performance metrics:

- Bias
- Confidence interval coverage
- Confidence interval width
- Mean squared error (MSE)

Simulation results are available as a CSV file under ``simulations/simulation_df.csv``.

Key Findings
------------

- **PMM** performs reliably under MCAR and mild MAR, particularly with symmetric distributions and large samples.
- **PMM** struggles under skewed distributions or structured missingness, often yielding biased estimates and reduced coverage.
- **midastouch** consistently matches or outperforms PMM in coverage and standard error estimation, especially under skewness or small sample sizes.
- Unlike PMM, **midastouch** requires no manual tuning of donor size ``k`` and, when combined with **HowManyImputations (HMI)**, provides an efficient and automated solution.

Getting Started
---------------

Installation
~~~~~~~~~~~~

Clone this repository:

.. code-block:: bash

   git clone https://github.com/Theanh2/Implementation_of_PMM_by_MICE.git@detached
   cd Implementation_of_PMM_by_MICE

Or directly install via pip:

.. code-block:: bash

   pip install git+https://github.com/Theanh2/Implementation_of_PMM_by_MICE.git@detached

.. toctree::
   :maxdepth: 2
   :caption: Contents

   modules

