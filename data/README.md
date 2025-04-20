# NHANES Datasets

This directory contains subsets of the NHANES (National Health and Nutrition Examination Survey) datasets, which are commonly used for demonstrating multiple imputation techniques. These datasets come built-in with R's MICE package.

## NHANES Dataset

The dataset (`nhanes.csv`) contains 25 observations with 4 variables:
- `age`: Age category (1, 2, or 3)
- `bmi`: Body Mass Index
- `hyp`: Hypertension status (1 or 2)
- `chl`: Cholesterol level

## NHANES2 Dataset

The NHANES2 dataset (`nhanes2.csv`) contains the exact same numerical values and missing data pattern as the nhanes. Key Difference is in variable types. In nhanes2, the variables age and hyp are categorical, while bmi and chl remain numeric. Age represents an ordered variable, and hyp represents a binary one.



## fdgs Dataset
The fdgs dataset (fdgs), which stands for "Fifth Dutch growth study 2009", contains age, height (hgt), weight (wgt), and region data for 10030 Dutch children.15 It also includes height and weight Z-scores (hgt.z, wgt.z). This dataset is notable for its larger size and has been used to demonstrate how MICE can address nonresponse in large-scale studies by creating synthetic cases through multiple imputation.
https://stefvanbuuren.name/fimd/fcs-and-jm.html 
https://amices.org/mice/reference/fdgs.html