from transformers import BartTokenizer, BartForConditionalGeneration

# Load the saved model
model_path = "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/summarizer"
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# Summarization function
def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=128,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)