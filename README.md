# UPI Fraud Detection System

## About the Project

This project is based on **Data Analysis using Python**.
The main goal is to analyze UPI transaction data and understand patterns of fraudulent activities.

In this project, I first performed data analysis and visualization to study the dataset, and then applied a machine learning model to predict fraud.

## Objective

* To analyze transaction data
* To identify patterns in fraudulent transactions
* To visualize important insights
* To build a model that predicts fraud

## What this project includes

### 1. Data Generation

A dataset of UPI transactions is generated with:

* Transaction amount
* Transaction hour
* Fraud status

### 2. Data Analysis

I performed **Exploratory Data Analysis (EDA)**, including:

* Statistical summary (mean, max, min)
* Fraud vs safe transaction count
* Correlation between features
* Group analysis (fraud by amount and time)

### 3. Data Visualization

The following graphs are used:

* Bar chart (Fraud vs Safe)
* Histogram (Amount distribution)
* Boxplot (Amount vs Fraud)
* Scatter plot (Hour vs Amount)
* Correlation matrix

### 4. Machine Learning

* Algorithm: Random Forest Classifier
* Purpose: Predict whether a transaction is fraud or safe

### 5. Model Evaluation

* Accuracy
* Confusion Matrix
* ROC Curve (AUC)
* Mean Squared Error

### 6. Interactive Dashboard

A web app is created using Streamlit where:

* User can input amount and time
* System predicts fraud in real-time
* Graphs and analysis are displayed

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit

## Project Structure

UPI_Fraud_Project/
│── main.py
│── app.py
│── data.csv
│── model.pkl
│── requirements.txt
│── README.md

## How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run data analysis and model

python main.py

### 3. Run the dashboard

streamlit run app.py

### 4. Open in browser

http://localhost:8501

## Key Insights

* Fraud transactions usually have higher amounts
* Fraud is more common during late night / early morning
* Patterns can be identified through data analysis

## Future Improvements

* Use real-world dataset
* Improve model accuracy
* Add more features like location or device info
* Deploy the system online

## Author

Rohan Das

## Final Note

This project helped me understand how data analysis and machine learning can be combined to solve real-world problems like fraud detection.
