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

The fdgs dataset (fdgs), which stands for "Fifth Dutch growth study 2009", contains age, height (hgt), weight (wgt), and region data for 10030 Dutch children. It also includes height and weight Z-scores (hgt.z, wgt.z). This dataset is notable for its larger size and has been used to demonstrate how MICE can address nonresponse in large-scale studies by creating synthetic cases through multiple imputation.

For more information, see:
- [FCS and JM](https://stefvanbuuren.name/fimd/fcs-and-jm.html)
- [MICE documentation](https://amices.org/mice/reference/fdgs.html)

## NHANES large Dataset

The nhanes large dataset is loaded to showcase working with data with larger size and much higher number of columns. 

NHANES data is collected in cycles, each cycle being 2 years. Cycles can be found at [NHANES Data Page](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Demographics). We chose to work with the 2017-2018 cycle. Each cycle has five publicly available data categories:
- Demographics (DEMO)
- Dietary (DIET)
- Examination (EXAM)
- Laboratory (LAB)
- Questionnaire (Q)

The tables available for each category can be seen using:
```R
library(nhanesA)
nhanesTables("DIET", 2017)
```

### Tables included in the dataset:

- BPX_J: Blood Pressure (EXAM)
- HEQ_J: Hepatitis (Q)
- CDQ_J: Cardiovascular Health (Q)
- PAQ_J: Physical Activity (Q)
- IMQ_J: Immunization (Q)
- INQ_J: Income (Q)
- ALQ_J: Alcohol Use (Q)
- TCHOL_J: Cholesterol - Total (LAB)
- BMX_J: Body Measures (EXAM)
- SMQ_J: Smoking - Cigarette Use (Q)
- FERTIN_J: Ferritin (LAB)
- FETIB_J: Iron Status - Serum (LAB)
- HEPA_J: Hepatitis A (LAB)
- HEPB_S_J: Hepatitis B: Surface Antibody (LAB)

The `nhanes_large_descriptions.json` file contains full variable names of each table.

Tables are merged with each other by using Respondent sequence numbers, resulting in size 9254 rows x 161 columns. A lot of columns have a high percent of missing data and should rather be excluded while imputing.

This dataset can be used for demonstrating that with larger datasets one shouldn't take an approach of just blindly fitting, but rather do analysis beforehand.