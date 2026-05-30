# Caravan Insurance Purchase Prediction

## Project Overview

This project uses machine learning classification to predict whether a customer is likely to purchase caravan insurance.

The goal is not only to build a predictive model, but also to evaluate the model from a business decision-making perspective. In marketing and customer targeting, the company wants to identify customers who are more likely to buy, instead of spending resources contacting customers who are unlikely to purchase.

This project focuses on an imbalanced classification problem, where only a small percentage of customers actually purchased caravan insurance. Because of this imbalance, overall accuracy can be misleading. The analysis therefore emphasizes confusion matrices, predicted buyers, and precision among predicted buyers.

## Business Question

How can an insurance company identify customers who are more likely to purchase caravan insurance and improve the efficiency of a targeted marketing campaign?

More specifically, the project asks:

* Can customer profile variables help identify likely buyers?
* How does K-Nearest Neighbors perform on an imbalanced insurance dataset?
* Why is accuracy not enough when the target class is rare?
* Which model is more useful for business targeting based on precision?

## Dataset

The project uses the `Caravan` dataset from the `ISLR2` R package.

The dataset contains customer-level information and a binary response variable indicating whether the customer purchased caravan insurance.

### Dataset Details

* Observations: 5,822 customers
* Predictor variables: 85 customer profile variables
* Response variable: `Purchase`
* Target classes:

  * `Yes`: Customer purchased caravan insurance
  * `No`: Customer did not purchase caravan insurance
* Purchase rate: approximately 6%

Because the dataset is available directly through the `ISLR2` package, no external data file is required.

## Why This Problem Matters

This is a realistic business analytics problem because many marketing datasets are imbalanced. In this case, most customers do not purchase the product.

A model that always predicts “No” can achieve high accuracy because the majority of customers are non-buyers. However, such a model has no business value because it does not identify any potential buyers.

For a targeted marketing campaign, the company is more interested in questions such as:

* Among the customers predicted as buyers, how many actually buy?
* How many customers would be targeted by the campaign?
* Is the model better than a simple baseline?
* Does the model help allocate marketing resources more efficiently?

## Methodology

The analysis follows a structured machine learning workflow.

### 1. Data Inspection

The script first loads the `Caravan` dataset and creates summary outputs showing:

* Number of observations
* Number of predictor variables
* Response variable
* Purchase class distribution

A bar chart is also created to show the imbalance between customers who purchased and did not purchase caravan insurance.

### 2. Data Preprocessing

K-Nearest Neighbors is a distance-based algorithm, so feature scaling is required.

All predictor variables are standardized so that each variable has a comparable scale. The response variable `Purchase` is excluded from scaling.

The script also saves a standardization check showing the mean and standard deviation of selected scaled variables.

### 3. Train/Test Split

The data is split into training and test sets using the setup commonly used for this dataset:

* Test set: first 1,000 observations
* Training set: remaining observations

The model is trained on the training data and evaluated on the test data.

### 4. Baseline Model

A baseline model is created by predicting `No` for every customer.

This baseline helps show why accuracy can be misleading. Since most customers do not buy caravan insurance, this model achieves a relatively low error rate but identifies zero buyers.

### 5. K-Nearest Neighbors Models

The project trains and evaluates KNN models using three different K values:

* K = 1
* K = 3
* K = 5

For each model, the script generates:

* Predictions on the test set
* Confusion matrix
* Error rate
* Accuracy
* Number of predicted buyers
* Number of true positives
* Precision among predicted buyers

### 6. Model Evaluation

Instead of relying only on accuracy, the project focuses on business-relevant evaluation metrics.

The most important metric is precision among predicted buyers:

Precision = True Buyers Predicted as Buyers / Total Customers Predicted as Buyers

This metric is useful because it tells the business how efficient a targeted marketing campaign would be.

## Outputs

Running the script automatically creates output folders and saves tables and figures.

### Tables

The following CSV files are saved in `outputs/tables/`:

* `01_dataset_overview.csv`
* `02_purchase_class_distribution.csv`
* `03_standardization_check.csv`
* `04_baseline_performance.csv`
* `05_knn_performance_comparison.csv`
* `06_knn_confusion_matrices.csv`
* `07_model_performance_with_baseline.csv`
* `08_business_interpretation.csv`

### Figures

The following figures are saved in `outputs/figures/`:

* `01_purchase_class_distribution.png`
* `02_precision_by_k.png`
* `03_predicted_buyers_by_k.png`
* `04_error_rate_comparison.png`
* `05_confusion_matrices.png`

## Repository Structure

```text
caravan-insurance-purchase-prediction/
│
├── scripts/
│   └── caravan_knn_analysis.R
│
├── outputs/
│   ├── figures/
│   └── tables/
│
└── README.md
```

## Tools and Packages Used

* R
* ISLR2
* caret
* tidyverse
* ggplot2
* dplyr

## How to Run the Project

1. Clone or download this repository.

2. Open the project in RStudio or another R environment.

3. Run the script:

```r
source("scripts/caravan_knn_analysis.R")
```

The script will:

* Install required packages if needed
* Load the dataset
* Standardize predictor variables
* Train KNN models
* Evaluate model performance
* Save output tables
* Save output figures

## Key Findings

The dataset is highly imbalanced, with only a small share of customers purchasing caravan insurance.

The baseline model has strong apparent accuracy because it predicts the majority class, but it is not useful for marketing because it does not identify any buyers.

KNN models are able to identify a small group of potential buyers. This is more useful from a business perspective because the company can focus marketing efforts on customers with a higher predicted likelihood of purchasing.

As K increases, the model generally becomes more conservative and predicts fewer customers as buyers.

Precision among predicted buyers is a more meaningful metric than overall accuracy for this business problem because it measures the quality of the targeted customer list.

## Business Interpretation

For an insurance company, the value of the model is not simply whether it predicts most customers correctly. Since most customers do not buy, a high-accuracy model may still fail to support marketing decisions.

The more relevant business question is whether the model can identify a smaller group of customers who are more likely to purchase. This supports better targeting, lower campaign waste, and more efficient use of marketing resources.

This project demonstrates how machine learning results should be interpreted in the context of the business problem, especially when working with imbalanced data.

## Skills Demonstrated

This project demonstrates skills relevant to business analyst and data analyst roles, including:

* Translating a business problem into an analytical question
* Working with imbalanced classification data
* Preparing data for machine learning
* Applying K-Nearest Neighbors classification
* Comparing model performance across parameters
* Using confusion matrices for evaluation
* Interpreting precision for business targeting
* Creating reproducible outputs with tables and charts
* Communicating analytical findings in business terms

## Conclusion

This project shows that overall accuracy is not always the best measure of model performance, especially when the outcome of interest is rare.

For customer targeting, precision among predicted buyers provides a more practical way to evaluate whether a model can support business decisions. The analysis highlights the importance of choosing evaluation metrics that match the real business objective.
