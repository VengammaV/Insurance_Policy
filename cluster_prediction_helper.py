import pandas as pd
#from joblib import load
import joblib

# Path to the saved model and its components
MODEL_PATH =  "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/kmeans/model_data.joblib"

# Load the model and its components
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
pca = model_data['pca']

def preprocess_input(input_dict):
    expected_columns = ['Age', 'Annual_Income', 'Policy_Count', 'Total_Premium_Paid', 'Claim_Frequency', 'Policy_Upgrades']

    #df = pd.DataFrame(0, columns=expected_columns, index=[0])
    df = pd.DataFrame([{col: input_dict.get(col, 0) for col in expected_columns}])

    for key, value in input_dict.items():
        if key == 'Age':
            df['Age'] = value
        elif key == 'Annual_Income':
            df['Annual_Income'] = value
        elif key == 'Policy_Count':
            df['Policy_Count'] = value
        elif key == 'Total_Premium_Paid':
            df['Total_Premium_Paid'] = value
        elif key == 'Claim_Frequency':
            df['Claim_Frequency'] = value
        elif key == 'Policy_Upgrades':
            df['Policy_Upgrades'] = value

    scaled = scaler.transform(df)
    reduced = pca.transform(scaled)
    return reduced

def clusterpredict(input_dict):
    input =  preprocess_input(input_dict)
    cluster = model.predict(input)[0]
    return cluster
    
