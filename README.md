# Rossmann Store Sales Prediction

## Project Overview
This project focuses on forecasting daily sales for Rossmann drug stores using historical sales data, store metadata, and time-based features.
The goal is to build a reliable time series forecasting model while following good data science practices:
- proper validation strategy
- feature engineering
- avoidance of data leakage
- clear evaluation and interpretation of results

## Dataset
Kaggle “Rossmann Store Sales” dataset containing daily sales, promotions, store metadata, and time-based features.
https://www.kaggle.com/competitions/rossmann-store-sales

## Approach
### 1 Exploratory Data Analysis (EDA)
- Sales trends over time
- Weekly and seasonal patterns
- Effect of promotions and holidays
- Differences across store types
EDA notebooks are located in the notebooks/ directoyr

### 2 Feature Engineering
- calendar features (day, week, month)
- lagged sales values
- rolling statistics
- promotion indicators
- store metadata

Special care was taken to avoid data leakage, especially when creating time-dependent features.

### 3 Modeling
Models were trained using a custtom group time-based train/validation split, not random splitting.
This ensures the model is evaluated on future data only, closely matching real-world forecasting conditions.
Modeling logic code implemented in the src/train directory

### 4 Evaluation Strategy
- Time series validation
- Baseline model comparison
- Evaluation on unseen future periods

The main evaluation metric is RMSE, chosen because it strongly penalizes large forecasting errors, which are costly in business settings.

## Results
- Models outperform naive baselines
- Captures seasonal patterns and promotion effects effectively
- Demonstrates stable performance on validation data

Detailed evaluation and results can be found in the evaluation notebooks.

## What I Learned
- Proper time-based validation is critical for forecasting problems
- Importance of target transformations in skewed regression problems
- Feature leakage can silently inflate results if not handled carefully
- Why metric choice matters as much as model choice

## Possible Improvements
- Hyperparameter optimization
- Additional lag and rolling window features
- Model ensembling
