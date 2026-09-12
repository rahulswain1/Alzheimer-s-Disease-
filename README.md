# Alzheimer's Disease Classification

## Project Overview

This project focuses on classifying Alzheimer's Disease using demographic, lifestyle, medical, clinical, cognitive, and functional patient information.

The objective is to identify important factors associated with Alzheimer's Disease and develop a machine learning model that can classify patients based on their risk profile.

## Business Objective

- Detect patterns and early indicators related to Alzheimer's Disease.
- Identify important factors associated with disease classification.
- Develop machine learning models to predict the diagnosis outcome.
- Support data-driven decision-making for early detection and patient management.

## Live Application

The trained Alzheimer's Disease classification model is deployed as an interactive Streamlit application.

**Live App:**  
https://alzheimer-sdiseaseclassification-2ktayqrk57qdezkkwnqtaa.streamlit.app/

## Dataset

The dataset contains information for 2,149 patients between the ages of 60 and 90.

The dataset includes:

- Demographic information
- Lifestyle factors
- Medical history
- Clinical measurements
- Cognitive and functional assessments
- Symptoms and behavioral indicators
- Alzheimer's Disease diagnosis

## Project Workflow

1. Data Understanding
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Feature Selection
5. Train-Test Split
6. Model Building
7. Model Evaluation
8. Hyperparameter Tuning
9. Feature Importance Analysis
10. Model Deployment using Streamlit

## Machine Learning Models

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

Random Forest was selected for further optimization because it provided the strongest overall performance among the initial models.

## Final Model

A Random Forest model was tuned using Grid Search.

Best parameters:

- Number of estimators: 200
- Maximum depth: 20
- Minimum samples split: 2

The tuned Random Forest achieved:

- Accuracy: 94.42%
- Precision: 94.44%
- Recall: 89.47%
- F1 Score: 91.89%

## Important Features

Feature importance analysis identified the following as the most influential features in the model:

- FunctionalAssessment
- ADL
- MMSE
- MemoryComplaints
- BehavioralProblems
- DietQuality
- PhysicalActivity
- SleepQuality
- CholesterolTriglycerides
- CholesterolHDL

These feature importance results indicate which variables contributed most to the model's predictions. They should not be interpreted as proof of medical causation.

## Deployment

The final model was deployed as an interactive Streamlit application.

The application allows users to enter patient information and obtain a machine learning-based prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Repository Contents

- `ALZHEIMER'S_DISEASE.ipynb` – Complete project notebook
- `alzheimers_disease_data.csv` – Dataset
- `alzheimers_rf_model.pkl` – Tuned Random Forest model
- `feature_names.pkl` – Model feature names
- `app.py` – Streamlit application
- `requirements.txt` – Required Python packages

## Project Outcome

The project demonstrates a complete end-to-end machine learning workflow, from exploratory data analysis and model development to hyperparameter tuning, feature importance analysis, and deployment as a Streamlit application.

## Disclaimer

This project is developed for educational and data science project purposes. The model is intended as a decision-support demonstration and should not be used as a substitute for professional medical diagnosis.
