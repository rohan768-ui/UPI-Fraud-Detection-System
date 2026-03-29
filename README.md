# UPI Fraud Detection System

# About the Project

This project is a simple UPI Fraud Detection System built using Python and machine learning.
The main idea is to analyze transaction details like amount and time, and predict whether a transaction is safe or fraudulent.

Instead of using a real dataset, I created a sample dataset to simulate real-world transactions and applied a machine learning model on top of it.

# What this project 

Generates transaction data (amount, time, fraud status)
Applies logic to detect risky transactions
Trains a machine learning model (Random Forest)
Predicts whether a transaction is safe or fraud
Shows results using a simple web interface
Displays graphs and performance metrics

# Features

User can enter amount and time to check a transaction
Fraud detection based on trained model
Accuracy and model evaluation
Confusion matrix and ROC curve
Graphs for data visualization
Option to download dataset

# Technologies Used

Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
# How to Run the Project

Install required libraries:
pip install -r requirements.txt
Run the main file (this will generate data and train the model):
python main.py
Run the web app:
streamlit run app.py
Open your browser and go to:
http://localhost:8501

# Project Structure

UPI_Fraud_Project/
│── main.py
│── app.py
│── data.csv
│── model.pkl
│── requirements.txt
│── README.md

# Model Details

The model used is Random Forest Classifier.
It takes inputs like:

Transaction amount
Whether the amount is high
Whether the transaction is at night
Other derived features

Based on these, it predicts whether the transaction is fraudulent or not.

# Future Improvements
Use real transaction data
Improve model accuracy
Add more features (like location, device info)
Deploy the project online

# Author

Rohan Das

# Final Note

This project helped me understand how machine learning can be used in real-life problems like fraud detection.
It combines data analysis, model building, and a simple user interface.