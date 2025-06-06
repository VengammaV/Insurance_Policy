import pandas as pd
import joblib
#from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Path to the saved model and its components
MODEL_PATH =  "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/Claim_Risk/model1_data.joblib"

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']

def preprocess_input(input_dict):
    expected_columns = ["Customer_Age", "Annual_Income", "Claim_History", "Fraudulent_Claim", "Premium_Amount",
    "Risk_Score", "Policy_Type_Health", "Policy_Type_Home", "Policy_Type_Life", "Policy_Type_Travel", "Gender_M"
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
        elif key == 'Risk_Score':
            df['Risk_Score'] = value
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
    
    df["Claim_Amount"] = None
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df.drop("Claim_Amount", axis='columns', inplace=True)
    return df   

def predict(input_dict):
    input_df =  preprocess_input(input_dict)
    dummy_input = np.zeros((1, scaler.n_features_in_))
    dummy_input[0, -1] = model.predict(input_df)[0]
    inverse = scaler.inverse_transform(dummy_input)
    claim_amount = inverse[0, -1]
    return int(claim_amount)
    #return input_df

'''
# Let's say the model gave you a scaled prediction
scaled_prediction = model.predict(input_df)[0]  # example: 0.23

# Step 1: Create a dummy row with same number of features (e.g., 12)
dummy_input = np.zeros((1, scaler.n_features_in_))  # shape (1, 12)

# Step 2: Put the predicted value in the correct column (e.g., last column)
dummy_input[0, -1] = scaled_prediction  # assuming 'Claim_Amount' was last during training

# Step 3: Inverse transform
inverse = scaler.inverse_transform(dummy_input)

# Step 4: Extract the original Claim_Amount
claim_amount = inverse[0, -1]  # again, from the last column
'''
    
        