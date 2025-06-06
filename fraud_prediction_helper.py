import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Path to the saved model and its components
MODEL_PATH =  "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/fraud/model_fraud_data.h5"
MODEL_METADATA = "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/fraud/model_metadata.joblib"

# Load the model and its components
model = load_model(MODEL_PATH)

model_metadata = joblib.load(MODEL_METADATA)
scaler = model_metadata['scaler']
cols_to_scale = model_metadata['cols_to_scale']

def preprocess_input(input_dict):
    expected_columns = [
    "claim_amount", "income", "suspicious_flag", "claim_processing_days", "claim_to_income_ratio",
    "claim_type_Business Interruption", "claim_type_Fire", "claim_type_Liability",
    "claim_type_Medical", "claim_type_Natural Disaster", "claim_type_Personal Injury",
    "claim_type_Property Damage", "claim_type_Theft", "claim_type_Travel"
    ]

    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    for key, value in input_dict.items():
        if key == 'claim_amount':
            df['claim_amount'] = value
        elif key == 'income':
            df['income'] = value
        elif key == 'suspicious_flag':
            df['suspicious_flag'] = value
        elif key == 'policy_date':
            df['policy_date'] = value
        elif key == 'claim_date':
            df['claim_date'] = value
        elif key == 'claim_type':
            if value == 'Business Interruption':
                df['claim_type_Business Interruption'] = 1
            elif value == 'Fire':
                df['claim_type_Fire'] = 1
            elif value == 'Liability':
                df['claim_type_Liability'] = 1
            elif value == 'Medical':
                df['claim_type_Medical'] = 1
            elif value == 'Natural Disaster':
                df['claim_type_Natural Disaster'] = 1
            elif value == 'Personal Injury':
                df['claim_type_Personal Injury'] = 1
            elif value == 'Property Damage':
                df['claim_type_Property Damage'] = 1
            elif value == 'Theft':
                df['claim_type_Theft'] = 1
            elif value == 'Travel':
                df['claim_type_Travel'] = 1
    df['claim_date'] = pd.to_datetime(df['claim_date'])
    df['policy_date'] = pd.to_datetime(df['policy_date'])
    df['claim_processing_days'] = (df['claim_date'] - df['policy_date']).dt.days
    df['claim_to_income_ratio'] = df['claim_amount']*100 / df['income']
    df.drop(['policy_date','claim_date'], axis = 1,inplace=True)
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df 

def fraudpredict(input_dict):
    input_df =  preprocess_input(input_dict)
    predict_prob = model.predict(input_df)
    prediction = (predict_prob >= 0.5).astype(int)
    return (prediction.flatten())
    



