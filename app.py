import streamlit as st
import joblib
from claim_prediction_helper import predict # type: ignore
from risk_prediction_helper import riskpredict # type: ignore
from fraud_prediction_helper import fraudpredict # type: ignore
from cluster_prediction_helper import clusterpredict # type: ignore
from sentiment_helper import predict_sentiment # type: ignore
from summary_helper import summarize # type: ignore
from translate_helper import translate_fr, translate_es # type: ignore
from datetime import date

# App title
st.set_page_config(page_title="Insurance AI Dashboard", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Insurance Claim Prediction & Risk Analysis",
        "Fraud Detection",
        "Customer Segmentation",
        "Sentiment Analysis",
        "Policy Translation & Summarization",
        "AI Assistant"
    ]
)

# Page 1: Insurance Claim Prediction & Risk Analysis
if app_mode == "Insurance Claim Prediction & Risk Analysis":
    st.title("🚑 Insurance Claim Prediction & Risk Analysis")
    st.write("### Provide customer policy details to predict claim: ###")
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(2)

    with row1[0]:
        Customer_Age = st.slider("Age", 10, 80, 30)
    with row1[1]:
        Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with row1[2]:
        Annual_Income = st.slider("Income", 25000, 200000, 25000)

    with row2[0]:
        Policy_Type = st.selectbox("Policy", ["Health", "Home", "Life","Travel","Auto"])
    with row2[1]:
        Premium_Amount = st.slider("Premium", 500 , 10000,500)
    with row2[2]:
        Risk_Score = st.selectbox("Risk", ["Low", "Medium", "High"])
    
    with row3[0]:
        Claim_History = st.number_input("No. of claims", 0, 5,1)
    with row3[1]:
        Fraudulent_Claim = st.number_input("Fraud:0 NotFraud:1", 0, 1)

    # Preprocess
    # Map categorical values (this matches encoding during training)
    Risk_Score_Map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    input_dict = {
        "Customer_Age": Customer_Age,
        "Annual_Income": Annual_Income,
        "Claim_History": Claim_History,
        "Fraudulent_Claim": Fraudulent_Claim,
        "Premium_Amount": Premium_Amount,   
        "Policy_Type": Policy_Type,
        "Risk_Score": Risk_Score_Map[Risk_Score],
        "Gender": Gender,
         }

    if st.button('Predict Claim Amount'):
        prediction = predict(input_dict)
        st.success(f"The Claim Price Prediction is: {prediction}")
        #st.dataframe(df.head())
    
    st.write("### Provide customer policy details to predict Risk Score: ###")

    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(2)

    with row1[0]:
        Customer_Age = st.slider("Customer Age", 10, 80, 30)
    with row1[1]:
        Gender = st.selectbox("Customer Gender", ["Male", "Female", "Other"])
    with row1[2]:
        Annual_Income = st.slider("Customer Income", 25000, 200000, 25000)

    with row2[0]:
        Policy_Type = st.selectbox("Policy Type", ["Health", "Home", "Life","Travel","Auto"])
    with row2[1]:
        Premium_Amount = st.slider("Premium Amount", 500 , 10000,500)
    with row2[2]:
        Claim_Amount = st.slider("Claim Amount", 5000, 300000, 10000)
    
    with row3[0]:
        Claim_History = st.number_input("Total No. of claims", 0, 5,1)
    with row3[1]:
        Fraudulent_Claim = st.number_input("NotFraud:1 Fraud 0", 0, 1)

    input_dict = {
        "Customer_Age": Customer_Age,
        "Annual_Income": Annual_Income,
        "Claim_History": Claim_History,
        "Fraudulent_Claim": Fraudulent_Claim,
        "Premium_Amount": Premium_Amount,   
        "Claim_Amount": Claim_Amount,
        "Policy_Type": Policy_Type,
        "Gender": Gender,
         }

    if st.button('Predict Risk Score'):
        prediction = riskpredict(input_dict)
        st.success(f"The Predicted Risk Score (0:Low, 1:Medium, 2:High) is: {prediction}")
        #st.dataframe(df.head())


# Page 2: Fraud Detection
elif app_mode == "Fraud Detection":
    st.title("🔍 Fraud Detection")
    st.write("### Provide insurance transaction data to detect potential fraud cases: ###")
    row1 = st.columns(3)
    row2 = st.columns(3)
    with row1[0]:
        claim_amount = st.slider("Claim Amount", 5000, 200000, 1000)
    with row1[1]:
        income = st.slider("Customer Income", 25000, 200000, 25000)
    with row1[2]:
        suspicious_flag = st.number_input("Not_Suspicious:0, Suspicious:1", 0, 1)

    with row2[0]:
        policy_date = st.date_input(
                    "Select policy start date",
                    value=date.today(),
                    min_value=date(2000, 1, 1),
                    max_value=date.today()
                    )
    with row2[1]:
        claim_date = st.date_input(
                    "Select claim date",
                    value=date.today(),
                    min_value=date(2000, 1, 1),
                    max_value=date.today()
                    )
    with row2[2]:
        claim_type = st.selectbox("Claim Type", ["Business Interruption", "Fire", "Liability","Medical","Natural Disaster","Personal Injury", "Property Damage","Theft","Travel"])

    input_dict = {
        "claim_amount": claim_amount,
        "income": income,
        "suspicious_flag": suspicious_flag,
        "policy_date": policy_date,
        "claim_date": claim_date,   
        "claim_type": claim_type,
         }    

    if st.button('Predict Fraud Score'):
        prediction = fraudpredict(input_dict)
        st.success(f"The Predicted Fraud Score (0:NotFraud, 1:Fraud) is: {prediction}")    
   
# Page 3: Customer Segmentation
elif app_mode == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.write("### Segment customers based on their demographics and behavior. ###")
    st.write('Each cluster reveals a customer segment, e.g.:')
    st.write('Cluster 2: Avg-Age:45, Avg-Income:162000, high policy count, high premium paid, more-policy upgrades group')
    st.write('Cluster 1: Avg-Age: 36, Avg-Income:45000, low-claims group')
    st.write('Cluster 0: Avg-Age:56, Avg-Income:65000, high claim-freq group')

    row1 = st.columns(3)
    row2 = st.columns(3)

    with row1[0]:
        age = st.slider("Age", 18, 100, 30)
    with row1[1]:
        annual_income = st.slider("Annual Income", 25000, 300000, 25000)
    with row1[2]:
        policy_count = st.number_input("Policy Count", min_value=1, max_value=5, value=1)

    with row2[0]:
        total_premium_paid = st.slider("Total Premium Paid", 500 , 20000 , 500)
    with row2[1]:
        claim_frequency = st.number_input("Total No. of claims", 0, 5 ,1)
    with row2[2]:
        policy_upgrades = st.number_input("Policy Upgrades", min_value=0, max_value=5, value=0)

    input_dict = {
        "Age": age,
        "Annual_Income": annual_income,
        "Policy_Count": policy_count,
        "Total_Premium_Paid": total_premium_paid,
        "Claim_Frequency": claim_frequency,
        "Policy_Upgrades": policy_upgrades
        }

    if st.button('Predict Customer Segment'):
        cluster_segment = clusterpredict(input_dict)
        st.success(f"The Predicted Customer Segment is : {cluster_segment}")
        #st.dataframe(prediction.head())

# Page 4: Sentiment Analysis
elif app_mode == "Sentiment Analysis":
    st.title("💬 Sentiment Analysis")
    st.write("Analyze customer feedback and reviews to understand sentiment.")

    text_input = st.text_area("Enter customer feedback")

    # Mapping labels to sentiment (customize if you used different labels)
    label_map = {0: "Negative", 1: "Neurtal", 2: "Positive"} 

    if st.button("Analyze Sentiment"):
        if text_input.strip() != "":
            # TODO: Load sentiment model and get sentiment
            label_id, probabilities = predict_sentiment(text_input)
            sentiment = label_map[label_id]
            st.success(f"**Predicted Sentiment:** {sentiment}")
            #st.write("**Class Probabilities:**")
            #for i, prob in enumerate(probabilities):
            #    st.write(f"{label_map.get(i, str(i))}: {prob:.4f}")
        else:
            st.error("Please enter some text.")

# Page 5: Policy Translation & Summarization
elif app_mode == "Policy Translation & Summarization":
    st.title("📄 Insurance Policy Translation & Summarization")
    st.subheader("Translate English policy documents to French/Spanish.")
    en_text = st.text_area("Enter Full Policy English Text", height=200)
    if st.button("Translate_French"):
        if en_text.strip():
            result = translate_fr(en_text)
            st.write("Policy details in French")
            st.success(result)
        else:
            st.warning("Please enter a policy text.")
    if st.button("Translate_Spanish"):
        if en_text.strip():
            result = translate_es(en_text)
            st.write("Policy details in Spanish:")
            st.success(result)
        else:
            st.warning("Please enter a policy text.") 
    
    st.subheader("Summarize policy documents.")
    input_text = st.text_area("Enter Full Policy Text", height=300)
    if st.button("Summarize"):
        if input_text.strip():
            result = summarize(input_text)
            st.subheader("Summary")
            st.success(result)
        else:
            st.warning("Please enter a policy text.")

# Page 6: Chatbot - AI Assitant
elif app_mode == "AI Assistant":
    st.title(" 🤖 Chatbot - AI Assistant")
    # chatbot_insurance.py

    import streamlit as st
    import pandas as pd
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from transformers import BartTokenizer, BartForConditionalGeneration

    # --------------------
    # STEP 1: Load Excel Data
    # --------------------
    @st.cache_data
    def load_data(file_path):
        df = pd.read_excel(file_path)
        df = df.dropna(subset=['Policy_Text_EN_Clean'])  # Ensure non-empty entries
        return df

    # --------------------
    # STEP 2: Embed Texts
    # --------------------
    @st.cache_resource
    def build_index(texts):
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embed_model.encode(texts, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings))
        return embed_model, index, embeddings

    # --------------------
    # STEP 3: Retrieve Top k Chunks
    # --------------------
    def retrieve(query, model, index, texts, k=3):
        q_emb = model.encode([query])
        _, topk = index.search(np.array(q_emb), k)
        return [texts[i] for i in topk[0]]

    # --------------------
    # STEP 4: Generate Answer using BART
    # --------------------
    @st.cache_resource
    def load_bart():
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        return tokenizer, model

    def generate_answer(context, query, tokenizer, model):
        input_text = f"Context: {context}\n\nQuestion: {query}"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        output = model.generate(**inputs, max_length=128)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # --------------------
    # Streamlit UI
    # --------------------
    st.title("📄 Insurance Policy Chatbot")

    excel_file_path = "C:/Users/Harish/Desktop/GUVI/FinalProject_2/Data/insurance_policies.xlsx"  # ✅ your file path here
    df = load_data(excel_file_path)
    policy_texts = df['Policy_Text_EN_Clean'].tolist()
    embed_model, faiss_index, _ = build_index(policy_texts)
    tokenizer, bart_model = load_bart()

    user_query = st.text_input("🧠 Ask a question:")
    if user_query:
        docs = retrieve(user_query, embed_model, faiss_index, policy_texts)
        combined_context = " ".join(docs)
        answer = generate_answer(combined_context, user_query, tokenizer, bart_model)
        st.markdown(f"**🤖 Answer:** {answer}")






