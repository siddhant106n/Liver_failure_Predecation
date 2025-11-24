import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

# --- PART 1: TRAINING ---
def train():
    print("\n[1/3] Loading dataset...")
    try:
        # Load the dataset
        data = pd.read_csv('data/indian_liver_patient.csv')
    except FileNotFoundError:
        print("❌ Error: 'indian_liver_patient.csv' not found in 'data/' folder.")
        sys.exit()

    # Data Preprocessing
    # Fill missing values
    data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean())

    # Encode Gender
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    
    # Handle Target Column (Adjust based on your CSV column name if different)
    if 'Dataset' in data.columns:
        # 1=Disease, 2=No Disease -> Convert to 1=Disease, 0=No Disease
        data['Dataset'] = data['Dataset'].apply(lambda x: 1 if x == 1 else 0)
        X = data.drop('Dataset', axis=1)
        y = data['Dataset']
    else:
        print("❌ Error: Target column 'Dataset' not found.")
        sys.exit()

    # Train
    print("[2/3] Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"✅ Model trained! Accuracy: {acc*100:.2f}%")

    # Return model and encoder so we can use them immediately
    return model, le

# --- PART 2: PREDICTION ---
def predict(model, le):
    print("\n" + "="*40)
    print(" LIVER FAILURE PREDICTION (TEST MODE) ")
    print("="*40)
    print("Enter patient details to test the model:\n")

    try:
        age = int(input("1. Age: "))
        gender_txt = input("2. Gender (Male/Female): ").strip().capitalize()
        tb = float(input("3. Total Bilirubin (e.g., 0.7): "))
        db = float(input("4. Direct Bilirubin (e.g., 0.1): "))
        alkphos = int(input("5. Alkaline Phosphotase (e.g., 187): "))
        sgpt = int(input("6. SGPT (e.g., 16): "))
        sgot = int(input("7. SGOT (e.g., 18): "))
        tp = float(input("8. Total Proteins (e.g., 6.8): "))
        alb = float(input("9. Albumin (e.g., 3.3): "))
        ag_ratio = float(input("10. A/G Ratio (e.g., 0.9): "))
        
        # Encode Gender input
        gender_val = 1 if gender_txt == 'Male' else 0

        # Create DataFrame for prediction
        input_data = pd.DataFrame([{
            'Age': age, 'Gender': gender_val, 'Total_Bilirubin': tb, 
            'Direct_Bilirubin': db, 'Alkaline_Phosphotase': alkphos, 
            'Alamine_Aminotransferase': sgpt, 'Aspartate_Aminotransferase': sgot, 
            'Total_Protiens': tp, 'Albumin': alb, 'Albumin_and_Globulin_Ratio': ag_ratio
        }])

        # Predict
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        print("\n--- RESULT ---")
        if pred == 1:
            print(f"⚠️ POSITIVE: High Risk of Liver Disease ({prob*100:.1f}%)")
        else:
            print(f"✅ NEGATIVE: Low Risk ({ (1-prob)*100:.1f}%)")
            
    except ValueError:
        print("Invalid input. Please enter numbers.")

if __name__ == "__main__":
    # Run everything at once
    trained_model, gender_encoder = train()
    predict(trained_model, gender_encoder)