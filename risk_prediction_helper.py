import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Path to the saved model and its components
MODEL_PATH =  "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/Claim_Risk/model2_data.joblib"

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
cols_to_scale = model_data['cols_to_scale']

def preprocess_input(input_dict):
    expected_columns = ["Customer_Age", "Annual_Income", "Claim_History", "Fraudulent_Claim", "Premium_Amount",
    "Claim_Amount", "Policy_Type_Health", "Policy_Type_Home", "Policy_Type_Life", "Policy_Type_Travel", "Gender_M"
       ]
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    for key, value in input_dict.items():
        if key == 'Customer_Age':
            df['Customer_Age'] = value
        elif key == 'Annual_Income':
            df['Annual_Income'] = value
        elif key == 'Claim_History':
            df['Claim_History'] = value
        elif key == 'Fraudulent_Claim':
            df['Fraudulent_Claim'] = value
        elif key == 'Premium_Amount':
            df['Premium_Amount'] = value
        elif key == 'Claim_Amount':
            df['Claim_Amount'] = value
        elif key == 'Policy_Type':
            if value == 'Health':
                df['Policy_Type_Health'] = 1
            elif value == 'Home':
                df["Policy_Type_Home"] = 1
            elif value == 'Life':
                df["Policy_Type_Life"] = 1
            elif value == 'Travel':
                df["Policy_Type_Travel"] = 1
        elif key == 'Gender':
            if value == 'Male':
                df['Gender_M'] = 1
    
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df   

def riskpredict(input_dict):
    input_df =  preprocess_input(input_dict)
    prediction = model.predict(input_df)
    return prediction