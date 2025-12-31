# 🏦 Bank Loan Approval Prediction System

This is a Machine Learning project that automates the loan eligibility process using a **Random Forest Classifier**. The model achieves **99.8% accuracy** and is deployed via **Streamlit**.

## 🚀 Live Demo
You can access the live web application here: [PASTE YOUR STREAMLIT LINK HERE]

## 📌 Project Overview
The goal of this project is to provide banks with an instant, data-driven tool to decide if a loan applicant is eligible. It reduces manual processing time and minimizes human bias.

### Key Features:
* **Predictive Power:** Instant "Approved" or "Rejected" status.
* **Feature Engineering:** Uses advanced ratios like Loan-to-Income and Asset-to-Loan.
* **High Accuracy:** Near-perfect prediction with zero False Negatives.

## 📊 Insights & Discoveries
* **CIBIL Score** is the most significant factor (79% importance).
* **Assets** (Residential & Bank) act as crucial secondary safety nets.
* **Random Forest** outperformed Logistic Regression by capturing complex, non-linear patterns in banking data.

## 🛠️ Tech Stack
* **Language:** Python
* **Model:** Random Forest Classifier
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib
* **Deployment:** Streamlit Cloud

## 📂 Project Structure
* `app.py`: The Streamlit web interface code.
* `loan_model.pkl`: The saved Random Forest model.
* `scaler.pkl`: The saved data scaler.
* `requirements.txt`: List of dependencies for deployment.
* `loan_prediction.ipynb`: The notebook showing full data analysis and model training.
* Data_dictionary :A Data Dictionary is a document that explains exactly what every column in your dataset represents.

---
**Developed by [Your Name]**