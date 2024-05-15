# customer-churn-prediction

This project is focused on predicting customer churn using a Random Forest classifier. The dataset used in this project is a typical customer churn dataset, with features related to customer demographics, account information, and service usage.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Logging](#logging)
- [License](#license)

## Overview

Customer churn prediction helps businesses to identify customers who are likely to leave the service. By predicting churn, companies can take proactive measures to retain customers and improve customer satisfaction.

This project includes:
- Data loading and cleaning
- Feature engineering
- Data preparation
- Model training using Random Forest with hyperparameter tuning
- Model evaluation

## Data

The dataset used in this project should be in CSV format and include the following columns:

- `customerID`: Unique identifier for each customer
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

The target variable is `Churn`, which indicates whether the customer has churned (Yes) or not (No).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/gabrieltonyy/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your data file is named `Customer-Churn.csv` and is placed in the root directory of the project.
2. Run the script:
    ```sh
    python churn.py
    ```

## Model Training and Evaluation

The model is trained using a Random Forest classifier. The script performs the following steps:

1. **Load and Clean Data**: Loads the CSV file, handles missing values and duplicates.
2. **Feature Engineering**: Currently no additional features are created.
3. **Prepare Data**: Encodes categorical features, scales numerical features, and splits the data into training and testing sets.
4. **Train Model**: Uses SMOTE to handle class imbalance and GridSearchCV for hyperparameter tuning.
5. **Evaluate Model**: Evaluates the model using accuracy, classification report, confusion matrix, and ROC AUC score.

The best hyperparameters and evaluation metrics are logged during the process.

## Logging

The script uses Python's built-in `logging` module to log information at various stages of the pipeline. Logs include:
- Data loading status
- Data cleaning steps
- Feature engineering steps
- Data preparation steps
- Model training status and best parameters
- Model evaluation metrics

