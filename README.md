# FraudShield Dashboard

**FraudShield** is an advanced Streamlit dashboard for **credit card fraud detection**.  
It integrates data preprocessing, EDA, feature importance visualization, and ML predictions in a user-friendly interface.


## Features

- **Data Exploration:** Visualize dataset distributions and correlations.  
- **ML Model Predictions:** Predict fraud probability for test transactions.  
- **Feature Importance:** Top features influencing model decisions displayed as charts and tables.  
- **Interactive Dashboard:** Built using Streamlit with responsive plots and tables.  
- **CSV & Notebook Support:** Easily run analysis and visualize results locally.

  ## Project Structure
  fraudshield/
│── app.py # Streamlit app
│── eda/ # Jupyter notebooks and feature exploration
│ └── Untitled.ipynb
│── data/ # CSV files (excluded from GitHub)
│── requirements.txt # Dependencies
│── .gitignore # Ignored files and folders

## ML Model
The dashboard uses a machine learning model trained on historical credit card transactions.
Predicts fraud probability with accuracy and highlights key features influencing predictions.

## Usage
Upload your credit card transaction CSV (optional if dataset is preloaded).
View fraud probability distribution and top 15 feature importance graphs.
Analyze model predictions interactively.

## Notes
Large datasets are not included in the repository to keep it lightweight.
Ensure dependencies in requirements.txt are installed before running.

## Author
Aditi Pathak
