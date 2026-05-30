# Caravan Insurance Purchase Prediction

## Project Overview

This project uses machine learning classification to predict whether customers are likely to purchase caravan insurance.

The analysis focuses on a common business analytics problem: identifying likely buyers in an imbalanced dataset, where only a small share of customers purchase the product.

## Business Question

How can an insurance company identify customers who are more likely to purchase caravan insurance, while avoiding inefficient targeting of customers who are unlikely to buy?

Rather than relying only on overall accuracy, this project evaluates model performance using confusion matrices and precision among predicted buyers, which is more relevant for marketing and customer targeting decisions.

## Dataset

The project uses the `Caravan` dataset from the `ISLR2` R package.

The dataset contains:
- 5,822 customer observations
- 85 predictor variables
- A binary response variable: `Purchase`
- Only about 6% of customers purchased caravan insurance

Because the dataset is available directly through the `ISLR2` package, no external data file is required.

## Methods Used

- Data inspection
- Feature standardization
- Train/test split
- K-Nearest Neighbors classification
- Model comparison for K = 1, 3, and 5
- Confusion matrix evaluation
- Precision among predicted buyers
- Business interpretation for imbalanced classification

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
├── README.md
```

## Outputs

Running the R script saves:

### Tables
- Dataset overview
- Class distribution
- Baseline model performance
- KNN model performance for K = 1, 3, and 5
- Confusion matrices
- Precision comparison

### Figures
- Purchase class imbalance chart
- Precision by K
- Number of predicted buyers by K
- Error rate comparison

## Tools Used

- R
- ISLR2
- caret
- ggplot2
- dplyr

## How to Run

1. Download or clone this repository.
2. Open R or RStudio.
3. Run:

```r
source("scripts/caravan_knn_analysis.R")
```

The script will load the dataset from `ISLR2` and save outputs inside the `outputs/` folder.

## Key Learning

Overall accuracy can be misleading when the target class is rare. In this case, a model that always predicts “No” can achieve high accuracy but provides little business value.

Precision among predicted buyers is more useful because it helps evaluate how efficient a targeted marketing campaign would be.
