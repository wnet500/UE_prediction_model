# UE_prediction_model
Development and Validation of Unplanned Extubation Prediction Models Using Intensive Care Unit Data: Comparative Machine Learning Study

## datasets
Here, the datasets we used in the research paper can not be released for personal information protection

Instead, you can identify the structure of datasets and their examples 
please refer to `features_details.csv` and `UE_sample_data.csv` in the datasets folder

> `features_details.csv` consists of features, description, type and details

> `UE_sample_data.csv` shows the examples of the datasets used for modeling (Note that this is not real patients' dataset)

## python_code

### 1. feature_selection
Codes for input normalizationm, and feature selection

### 2. param_search_and_develop_models
Codes about parameter tuning for random forest, support vector machine, logistic regression, and artificial neural network

### 3. validation
Codes for theshold decision, internal validation, and performance evaluation in the validation sets

### 4. calibraion_plot
Codes for plotting calibration curve, and calculating brier score to evaluate calibration

### 5. net_benefit
Codes for decision curve to check clinical usefulness of the model

## R_code

### 1. calibraion_evaluation
Codes for The Integrated Calibration Index (ICI) and the Hosmer-Lemeshow goodness-of-fit statistic to evaluate calibration

## 
*** Note that all codes are executable ONLY if your own data exist
