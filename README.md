# Rossmann Store Sales Prediction

## Project Overview
Predict daily store sales using historical data to better understand the impact of promotions, store types, and temporal patterns.

## Dataset
Kaggle “Rossmann Store Sales” dataset containing daily sales, promotions, store metadata, and time-based features.

## Approach
- Exploratory data analysis to understand sales distributions and store behavior
- Feature engineering based on promotions, store types, and temporal effects
- Custom time-based cross-validation to avoid data leakage
- Models evaluated using RMSE on Sales to handle skewness

## Results
- Models outperform naive baselines
- Promotion and store type interactions are key drivers
- Error distribution is more stable after target transformation

## What I Learned
- How to design **custom time-based cross-validation** for temporal data
- Importance of target transformations in skewed regression problems
- Why business logic should guide feature engineering
- Why metric choice matters as much as model choice

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib

## Author
**Martin Harutyunyan**  
GitHub: https://github.com/themainpie
