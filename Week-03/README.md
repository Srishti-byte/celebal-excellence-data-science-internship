# Week 3 – Country Intelligence System using Machine Learning

## Objective

The objective of this project is to analyze country-level socio-economic data using machine learning techniques to identify meaningful clusters and build classification models capable of predicting those clusters.

---

## Dataset

The dataset contains socio-economic indicators such as:

- Child Mortality
- Income
- GDP per Capita
- Life Expectancy
- Inflation
- Exports
- Imports
- Health Expenditure
- Total Fertility Rate

---

## Tasks Performed

- Data preprocessing and feature scaling
- K-Means clustering using the Elbow Method
- Cluster evaluation using Silhouette Score
- PCA for cluster visualization
- Experimental comparison using K = 3 and K = 5
- DBSCAN clustering and comparison with K-Means
- Cluster profiling and interpretation
- Random Forest classification
- XGBoost classification
- Model comparison
- Feature importance analysis

---

## Results

- K-Means successfully segmented countries based on socio-economic indicators.
- PCA provided a clear visualization of the generated clusters.
- DBSCAN was evaluated as an alternative clustering algorithm.
- Random Forest achieved **97.1% accuracy**.
- XGBoost achieved **100% accuracy**, making it the best-performing model.
- GDP per Capita, Child Mortality, and Income were identified as the most influential features.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Google Colab

---

## Files

- `Week3_SrishtiGupta.ipynb` – Complete implementation of the project.
- `country_data.csv` – Dataset used for analysis.
