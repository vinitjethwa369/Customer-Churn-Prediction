# Customer-Churn-Prediction

This repository contains code and resources for predicting customer churn in a banking scenario using machine learning techniques. Churn prediction is a crucial task for businesses to retain customers and optimize their strategies. In this project, we explore various machine learning algorithms and techniques to predict whether a customer is likely to leave the bank.

# Table of Contents
- Introduction
- Dataset
- Data Preprocessing
- Modeling
- Evaluation
- Conclusion
- Usage
  
# Introduction
Customer churn refers to the phenomenon where customers cease their relationship with a company or business. It's a critical metric for businesses as retaining existing customers is often more cost-effective than acquiring new ones. Predicting churn allows businesses to proactively take actions to retain customers and improve customer satisfaction.

In this project, we aim to predict customer churn for a bank using machine learning algorithms. We explore different models and techniques to build an accurate churn prediction model.

# Dataset
The dataset used in this project is the "Churn_Modelling.csv" file, which contains various features related to bank customers such as their credit score, age, balance, etc. The target variable is "Exited," which indicates whether a customer has churned or not (1 for churned, 0 for not churned).

# Data Preprocessing
Loaded the dataset using pandas.
Dropped irrelevant columns like "RowNumber," "CustomerId," and "Surname."
Encoded categorical variables using one-hot encoding.
Handled imbalanced data using Synthetic Minority Over-sampling Technique (SMOTE).
Split the dataset into training and testing sets.
Performed feature scaling using StandardScaler.

# Modeling
We experimented with several classification algorithms:

- Logistic Regression
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Evaluation
We evaluated the performance of each model using various metrics such as accuracy, precision, recall, and F1-score. Additionally, we visualized the distribution of the target variable and the accuracy of different models using bar plots.

# Conclusion
SMOTE significantly improved the performance of the models by balancing the target classes.
Gradient Boosting Classifier achieved the highest accuracy among all models.
We saved the trained Random Forest model using joblib for future use.

# Usage
To use the churn prediction model:

Clone this repository to your local machine.
Ensure you have the necessary dependencies installed (numpy, pandas, scikit-learn, imbalanced-learn, matplotlib, seaborn).
Run the provided Jupyter Notebook or Python script to train and evaluate the models.
Once trained, you can use the saved model for predicting churn on new data.
Feel free to explore and modify the code according to your requirements.

Happy predicting!
