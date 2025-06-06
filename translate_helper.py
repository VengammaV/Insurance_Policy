from transformers import MarianMTModel, MarianTokenizer
import torch

# Load French fine-tuned model and tokenizer
model_path = "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/frenchtranslator"
tokenizer = MarianTokenizer.from_pretrained(model_path)
fr_model = MarianMTModel.from_pretrained(model_path).to("cpu")

# Load Spanish fine-tuned model and tokenizer
model_path = "C:/Users/Harish/Desktop/GUVI/FinalProject_2/artifacts/spanishtranslator"
es_model = MarianMTModel.from_pretrained(model_path).to("cpu")

# Set the language codes
tokenizer.src_lang = "en_XX"
fr_tgt_lang = "fr_XX"
es_tgt_lang = "es_XX"

def translate_fr(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = fr_model.generate(**inputs, max_length=128)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def translate_es(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = es_model.generate(**inputs, max_length=128)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
