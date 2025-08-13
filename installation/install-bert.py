# script/download_bert.py
from transformers import BertForQuestionAnswering, BertTokenizer
import os

# Define the model name from Hugging Face Model Hub
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"

# Define the directory where the model and tokenizer will be saved
# This will be created inside the current working directory from where the script is run
output_dir = "./bert_model"

print(f"Attempting to download and save model: {model_name}")
print(f"Model and tokenizer will be saved to: {os.path.abspath(output_dir)}")

try:
    # Load the model and tokenizer from Hugging Face
    # These commands will trigger the download
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Save the model and tokenizer to the specified local directory
    # This will create the 'bert_model' directory if it doesn't exist
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nSuccessfully downloaded and saved '{model_name}' to '{output_dir}'")
    print("You can now use this model offline!")

except Exception as e:
    print(f"\nAn error occurred during download or saving: {e}")
    print("Please check the following:")
    print("1. Your internet connection.")
    print(f"2. The model name: '{model_name}' (ensure it's typed correctly).")
    print("3. Sufficient disk space (this model is several hundred MBs).")
    print("4. If you encountered a 'ModuleNotFoundError', ensure your virtual environment is active and 'transformers' and 'torch' are installed.")

