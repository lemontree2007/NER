from flask import Flask, render_template, request
from PyPDF2 import PdfReader
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import pandas as pd
import streamlit as st

app = Flask(__name__)

# Load configuration, model, and tokenizer
model = AutoModelForTokenClassification.from_pretrained("./model/ner-model")
tokenizer = AutoTokenizer.from_pretrained("./model/ner-tokenizer")

# Label mapping dictionary
label_dict = {
    0: 'O',  
    1: 'B-per',  
    2: 'I-per',  
    3: 'B-geo',  
    4: 'I-geo',  
    5: 'B-org',  
    6: 'I-org',  
    7: 'B-tim',  
    8: 'I-tim',  
    9: 'B-eve',  
    10: 'I-eve',  
    11: 'B-gpe',  
    12: 'I-gpe',  
    13: 'Miscellaneous',  
}

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def merge_subword_tokens(tokens, labels):
    merged_tokens = []
    merged_labels = []
    current_token = ""
    current_label = ""

    for token, label in zip(tokens, labels):
        if token.startswith("##"):
            current_token += token[2:]
        else:
            if current_token:
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token
            current_label = label
    
    if current_token:
        merged_tokens.append(current_token)
        merged_labels.append(current_label)

    return merged_tokens, merged_labels

def filter_and_correct_labels(tokens, labels):
    corrected_labels = []
    for token, label in zip(tokens, labels):
        if token == '.':
            corrected_labels.append((token, 'O'))
        elif label.startswith('I-'):
            corrected_labels.append((token, label))
        else:
            corrected_labels.append((token, 'O'))
    return [label for _, label in corrected_labels]

def predict_text(text):
    try:
        # Tokenize the input text
        inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
        # Set the model to evaluation mode
        model.eval()

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predictions (logits)
        predictions = outputs.logits.argmax(dim=-1)

        # Load the dataset
        data = pd.read_csv('./dataset/ner_dataset.csv', encoding="latin1")

        # Fill missing values in the 'Sentence #' column
        data['Sentence #'] = data['Sentence #'].ffill()

        # Ensure that the Word column is a string
        data['Word'] = data['Word'].astype(str)

        # Get the unique labels
        unique_labels = set(data['Tag'].values)

        # Create a dictionary mapping labels to integers
        label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}

        # Map IDs back to labels
        id_to_label = {v: k for k, v in label_to_id.items()}

        # Convert predictions to labels
        predicted_labels = [id_to_label[p.item()] for p in predictions[0]]

        # Pair the tokens with their predicted labels
        predicted_entities = list(zip(inputs.tokens(), predicted_labels))

        print(f"Predicted entities: {predicted_entities}")

        # Group results by entity type
        grouped_results = {
            "Full Name": [],
            "Location": [],
            "Time": [],
            "Organisation": []
        }

        # Map the entity types to categories
        entity_map = {
            'B-per': 'Full Name',
            'I-per': 'Full Name',
            'B-geo': 'Location',
            'I-geo': 'Location',
            'B-org': 'Organisation',
            'I-org': 'Organisation',
            'B-tim': 'Time',
            'I-tim': 'Time'
            # Add other mappings as needed
        }

        # Populate the grouped results with linking logic
        current_entity = ""
        current_category = None

        for token, label in predicted_entities:
            if label in entity_map:
                category = entity_map[label]
                
                # Filter out unwanted tokens
                if (label == 'I-per' and token.lower() == 'he') or (label in ['B-tim', 'I-tim'] and token == '.'):
                    continue  # Skip unwanted tokens
                
                # If continuing an entity, append the token
                if current_category == category and not token.startswith("##"):
                    current_entity += f" {token}"
                elif token.startswith("##"):
                    # Append subword tokens without space
                    current_entity += token[2:]
                else:
                    # Add the completed entity to the group
                    if current_entity:
                        grouped_results[current_category].append(current_entity)

                    # Start a new entity
                    current_entity = token
                    current_category = category
            else:
                # Add the final entity if we encounter an 'O' label or another non-matching case
                if current_entity:
                    grouped_results[current_category].append(current_entity)
                    current_entity = ""
                    current_category = None

        # Add any remaining entity after the loop
        if current_entity:
            grouped_results[current_category].append(current_entity)

        return grouped_results

    except Exception as e:
        print(f"Prediction error: {e}")
        return []

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        pdf_text = ""
        text_input = ""
        predictions = None

        if request.method == 'POST':
            # Handle file upload
            uploaded_file = request.files.get('pdf_file')
            text_input = request.form.get('text_input', "")
            print(f"text_input file: {text_input}")

            # Check for file upload
            if uploaded_file and uploaded_file.filename != '':
                # Extract text from the uploaded PDF
                pdf_text = extract_text_from_pdf(uploaded_file)
                text_input = pdf_text  # Set extracted text as input for the textarea
                print(f"Extracted text: {pdf_text[:500]}...")  # Print first 500 chars of the text for debugging
                predictions = predict_text(pdf_text)
                print(f"Predictions: {predictions}")
            
            # Check for text input
            elif text_input.strip():
                print(f"Entered text: {text_input[:500]}...")  # Print first 500 chars of the text for debugging
                predictions = predict_text(text_input)
                print(f"Predictions: {predictions}")

            else:
                # If no file or text input, log the issue
                print("No file uploaded or text entered.")

        return render_template('index.html', pdf_text=pdf_text, text_input=text_input, predictions=predictions)
    except Exception as e:
        print(f"Error: {e}")
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
