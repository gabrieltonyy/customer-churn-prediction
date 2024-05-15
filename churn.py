import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        
        # Print the column names to check the dataset structure
        logging.info(f"Column names: {data.columns.tolist()}")
        
        # Handle missing values
        data.ffill(inplace=True)
        logging.info("Missing values handled.")
        
        # Remove duplicates
        data.drop_duplicates(inplace=True)
        logging.info("Duplicates removed.")
        
        return data
    except Exception as e:
        logging.error(f"Error loading or cleaning data: {e}")
        return None

def feature_engineering(data):
    try:
        # No new features to create for now, as we do not have a 'subscription_date' column which could be used to create new features that might be useful for the model.
        logging.info("No new features created.")
        
        return data
    except Exception as e:
        logging.error(f"Error in feature engineering: {e}")
        return None

# pre-process step of the pipeline to prepare the data for modeling
def prepare_data(data):
    try:
        # Define categorical and numerical features
        categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                                'PaperlessBilling', 'PaymentMethod']
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        # Convert TotalCharges to numeric, setting errors='coerce' to handle non-numeric values
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        
        # Handle any remaining missing values in TotalCharges
        data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
        
        # Encode categorical features and scale numerical features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # handle_unknown='ignore' to ignore unknown categories in the test set.
            ])
        
        X = data.drop(['customerID', 'Churn'], axis=1)
        y = data['Churn'].map({'Yes': 1, 'No': 0})
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training, 20% testing 
        logging.info("Data split into training and testing sets.")
        
        return X_train, X_test, y_train, y_test, preprocessor
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return None, None, None, None, None

def train_model(X_train, y_train, preprocessor):
    try:
        # Build the imbalanced pipeline
        imb_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Define parameter grid for GridSearchCV
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 10, None]
        }
        
        # GridSearchCV to find the best hyperparameters
        grid_search = GridSearchCV(estimator=imb_pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        logging.info("Model training complete.")
        
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")
        
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        logging.info(f"Model Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{classification_rep}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")
        logging.info(f"ROC AUC Score: {roc_auc}")
        
        return accuracy, classification_rep, conf_matrix, roc_auc
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        return None, None, None, None

# Main function to run the entire pipeline. acts as clean up code.
if __name__ == "__main__":
    # Path to the CSV file
    file_path = 'Customer-Churn.csv'
    
    # Load and clean the data
    data = load_and_clean_data(file_path)
    
    if data is not None:
        # Perform feature engineering
        data = feature_engineering(data)
        
        if data is not None:
            # Prepare data for modeling
            X_train, X_test, y_train, y_test, preprocessor = prepare_data(data)
            
            if X_train is not None and X_test is not None:
                # Train the Random Forest model
                model = train_model(X_train, y_train, preprocessor)
                
                if model is not None:
                    # Evaluate the model
                    evaluate_model(model, X_test, y_test)
