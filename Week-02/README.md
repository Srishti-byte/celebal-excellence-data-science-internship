# Week 2 - Tesla Deliveries Analysis and Forecasting

This repository contains my Week 2 assignment for the Celebal Technologies Data Science Internship. The notebook demonstrates a complete machine learning workflow, starting from data exploration and preprocessing to model building, evaluation, and time series analysis.

## Dataset

- **Dataset:** `tesla_deliveries_dataset_2015_2025.csv`
- Contains Tesla delivery-related information from 2015–2025.

## Tasks Performed

- Explored and understood the dataset
- Checked missing values and duplicate records
- Performed Exploratory Data Analysis (EDA)
- Encoded categorical features using LabelEncoder
- Created lag and rolling mean features
- Performed chronological train-test split
- Built and evaluated a Linear Regression model
- Performed 5-Fold Cross Validation
- Tuned a Random Forest Regressor using GridSearchCV
- Analyzed feature importance
- Performed the Augmented Dickey-Fuller (ADF) test
- Created a forecast table comparing actual and predicted values
- Compared the performance of both models

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## Files

```
Week-02/
├── week2_SrishtiGupta.ipynb
├── tesla_deliveries_dataset_2015_2025.csv
└── README.md
```

## Results

- Both Linear Regression and Random Forest achieved strong predictive performance with R² scores close to 0.99.
- Feature importance analysis identified **Production_Units** as the most influential feature.
- The ADF test confirmed that the target time series is stationary.

---

**Author:** Srishti Gupta
