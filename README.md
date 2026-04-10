# Ventilator-Prediction-Model
Machine Learning for Healthcare Resource Planning

This project uses Machine Learning regression to predict the number of ventilators available in a hospital based on critical operational factors such as:

Number of patients in ICU

Number of doctors available

The goal is to support healthcare resource planning and decision-making, especially during high-demand situations like pandemics or emergencies.

# Project Overview

Healthcare systems often face shortages of critical equipment. This project demonstrates how data-driven models can assist hospitals in:

Predicting ventilator availability

Understanding key influencing factors

Visualizing healthcare resource patterns

Supporting operational decisions

The model is built using a Random Forest Regressor with proper preprocessing and evaluation.

# Machine Learning Pipeline
Load Healthcare Dataset
        ↓
Data Cleaning & Column Renaming
        ↓
Feature Selection
        ↓
Train-Test Split (80/20)
        ↓
Pipeline:
  - StandardScaler
  - RandomForestRegressor
        ↓
Model Training
        ↓
Performance Evaluation
        ↓
Data Visualization & Analysis

# Dataset Description

The dataset (Ventilator model.csv) contains hospital operational data:

Feature	Description
Patients_in_ICU	Number of patients requiring intensive care
Doctors_Available	Available medical staff
Ventilators_Available	Target variable (number of ventilators)

# Model & Tools Used

scikit-learn – Model building & evaluation

Random Forest Regressor – Prediction model

pandas – Data handling

matplotlib – Visualization

seaborn – Advanced plots

# Model Architecture

A Pipeline is used to ensure clean and reusable preprocessing:

StandardScaler → RandomForestRegressor (100 Trees)
Why Random Forest?

Handles non-linear relationships

Robust to noise

Reduces overfitting via ensemble learning

Works well with small-to-medium datasets

# Model Evaluation

Metrics used:

Mean Squared Error (MSE)

R² Score (Coefficient of Determination)

Evaluation is performed on both:

Training data

Testing data

This helps verify generalization and detect overfitting.

# Rule-Based Decision Logic

In addition to ML predictions, a simple rule-based check is included:

If ICU patients < 100 → Ventilators available = 100 − Patients
Else → No ventilators available

This simulates emergency fallback logic for real-time decision-making.

# Data Visualization & Analysis

The project includes extensive visual analysis:

1. Correlation Heatmap

Shows relationships between ICU patients, doctors, and ventilators

2. Target Distribution

Distribution of ventilators available

3. Feature vs Target Relationships

Patients in ICU vs Ventilators

Doctors available vs Ventilators

4. Residual Analysis

Compares training and testing errors

Helps diagnose bias or variance issues

5. Feature Importance

Identifies which features most influence ventilator availability

# Installation & Usage
git clone https://github.com/yourusername/ventilator-prediction.git
cd ventilator-prediction
pip install -r requirements.txt
python main.py

# Requirements
pandas
scikit-learn
matplotlib
seaborn

# Key Concepts Demonstrated

Regression modeling

Ensemble learning (Random Forest)

Feature scaling using pipelines

Healthcare data analytics

Model evaluation (MSE, R²)

Residual analysis

Feature importance interpretation

Data visualization for decision support

# Potential Improvements

Add time-series forecasting (ICU trends)

Include hospital capacity constraints

Add more features (beds, oxygen supply, nurses)

Hyperparameter tuning (GridSearchCV)

Deploy as a Streamlit dashboard

Integrate real-time hospital data

Convert to REST API (FastAPI)

# Real-World Applications

Hospital resource management systems

Pandemic preparedness planning

Emergency response optimization

Government healthcare analytics

Smart hospital dashboards
