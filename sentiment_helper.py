from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
MODEL_DIR = "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/saved_bert_model"
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

# Mapping labels to sentiment (customize if you used different labels)
label_map = {0: "Negative", 1: "Neurtal", 2: "Positive"} 

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id, torch.softmax(logits, dim=1).squeeze().tolist()

