💳 Financial Fraud Detection System
Using GAN, XGBoost & SMOTE
📌 Project Description

This project implements a financial fraud detection system using machine learning and deep learning techniques. Due to the highly imbalanced nature of fraud datasets, techniques such as SMOTE and Generative Adversarial Networks (GANs) are used to enhance minority class representation and improve model performance.

The system preprocesses transaction data, balances the dataset, trains models, and evaluates fraud detection effectiveness using standard metrics.

🎯 Objectives

Detect fraudulent financial transactions

Handle class imbalance effectively

Improve recall and F1-score for fraud detection

Compare traditional ML and GAN-enhanced models

🧠 Methods & Algorithms

SMOTE (Synthetic Minority Oversampling Technique)

GAN for synthetic fraud data generation

XGBoost Classifier

Feature encoding and scaling

🗂 Dataset Information

Format: CSV

Target column: is_fraud

Contains numerical and categorical transaction features

Highly imbalanced real-world financial data

⚙️ Project Workflow

Load and clean data

Encode categorical features

Scale numerical features

Handle imbalance using SMOTE

Generate synthetic data using GAN

Train XGBoost and GAN-enhanced models

Evaluate model performance

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC-AUC Score

🚀 Results

Improved fraud detection after applying SMOTE and GAN

Better recall and F1-score compared to baseline models

Strong generalization on test data

🛠️ Technologies Used

Python

PyTorch

Scikit-learn

XGBoost

Pandas, NumPy

Matplotlib
