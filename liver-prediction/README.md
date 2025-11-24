🏥 Liver Failure Prediction System

📌 Project Overview

This project is a Machine Learning application designed to predict the likelihood of liver disease in patients based on clinical records. Using the Indian Liver Patient Records (ILPR) dataset, the system employs a Random Forest Classifier to analyze key health metrics (such as age, gender, bilirubin levels, and proteins) and classify patients into "High Risk" or "Low Risk" categories.

The solution is built as a Command Line Interface (CLI) tool, focusing on backend logic, efficient data processing, and accurate classification without the need for complex web frameworks.

✨ Features

Data Preprocessing: Automatically handles missing values (imputation) and encodes categorical data (Gender).

Model Training: Trains a robust Random Forest Classifier on historical medical data.

Model Persistence: Saves the trained model and encoders to disk (.pkl files) for reusability.

Real-time Prediction: Interactive terminal interface that accepts new patient data and provides an instant diagnosis with a confidence score.

User-Friendly CLI: Simple text-based prompts guide the user through the data entry process.

🛠️ Technologies Used

Language: Python 3.9+

Libraries:

pandas (Data Manipulation)

numpy (Numerical Computations)

scikit-learn (Machine Learning Model & Preprocessing)

joblib (Model Saving/Loading)

Tools: VS Code, Git/GitHub

📂 Project Structure

liver-failure-prediction/
├── data/
│   └── indian_liver_patient.csv   # Raw dataset
├── models/
│   ├── random_forest_model.pkl    # Trained model artifact
│   └── gender_encoder.pkl         # Saved LabelEncoder
├── src/
│   └── model_training.py          # Main script (Training + Prediction Logic)
├── requirements.txt               # Dependencies
├── README.md                      # Project Documentation
└── statement.md                   # Problem Statement & Scope


🚀 Steps to Install & Run

1. Prerequisites

Ensure you have Python installed. You can check this by running:

python --version


2. Clone the Repository

git clone [https://github.com/YOUR_USERNAME/liver-failure-prediction.git](https://github.com/YOUR_USERNAME/liver-failure-prediction.git)
cd liver-failure-prediction


3. Set Up Virtual Environment

# Create virtual environment
python3 -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate


4. Install Dependencies

pip install -r requirements.txt


5. Run the Application

This single script handles both training the model (if not already trained) and running the prediction interface.

python src/model_training.py


🧪 Instructions for Testing

Run the script command above.

The system will first output the Model Accuracy (e.g., ~75-80%).

It will then prompt you to enter patient details.

Test Case (High Risk):

Age: 65

Gender: Female

Total Bilirubin: 0.7

Direct Bilirubin: 0.1

Alkphos: 187

SGPT: 16

SGOT: 18

Total Proteins: 6.8

Albumin: 3.3

A/G Ratio: 0.9

Observe the output classification (Positive/Negative) and probability score.

📊 Results & Screenshots

(Add screenshots of your terminal running the code here)

Model Accuracy: ~75% (varies slightly based on random seed)

Algorithm: Random Forest Classifier (n_estimators=100)