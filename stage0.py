import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime

# Step 1: Read the csv file


def load_data_from_csv(file_path):
    try:
        # Try reading with utf-8 encoding
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback to ISO-8859-1 if utf-8 fails
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df[['question', 'human_answer', 'file_name']]

# Step 2: Load the content of the file represented by file_name


def load_file_content(file_name):
    file_dir = 'C:\\Users\\14107\\Desktop\\Courses\\COSC760BigData\\Project\\archive\\acquired-individual-transcripts\\acquired-individual-transcripts\\'
    with open(file_dir + file_name + ".txt", 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

# Step 3: Send context and question to Ollama


def query_ollama(context, question):
    ollama_api_url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        # "context": context,  # Ensure this is formatted correctly
        "prompt": question,
        "model": "rag3.2:3b",  # Adjust based on your Ollama model
        # "model": "llama3.1:8brag",  # Adjust based on your Ollama model\
        "stream": False
    }

    try:
        response = requests.post(ollama_api_url, json=payload, headers=headers)
        # # print(f"Response Status Code: {response.status_code}")
        # print(f"Response Text: {response.text}")
        # # response.raise_for_status()  # Check for 4xx or 5xx errors
        # data = response.json()
        # print(f"Text: {data.get("response")}")
        # # return data.get("text")

        if response.status_code == 200:
            return response.json().get("response")  # Ensure you're extracting the right key
        else:
            print(f"Error querying Ollama: {response.status_code} - {response.text}")
            return None  # Return None if there's an error
    except Exception as e:
        print(f"Exception occurred while querying Ollama: {e}")
        return None  # Return None in case of an exception

# Step 4: Generate embeddings and calculate cosine similarity
def calculate_cosine_similarity(text1, text2, model):
    embeddings1 = model.encode([text1], convert_to_tensor=True)
    embeddings2 = model.encode([text2], convert_to_tensor=True)
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

def main(csv_file):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight SBERT model
    
    # Load data from csv
    data = load_data_from_csv(csv_file)
    
    for index, row in data.iterrows():
        question = row['question']
        human_answer = row['human_answer']
        file_name = row['file_name']
        
        # Load the content of the file
        file_content = load_file_content(file_name)
        # print(f"File content for {file_name}: ", file_content)
        
        # Query Ollama
        generated_answer = query_ollama("", question)
        # generated_answer = query_ollama(file_content, question)

        if generated_answer is None:
            print(f"Skipping row {index + 1} due to API error.")
            continue  # Skip further processing for this row if no valid answer
        
        # Calculate cosine similarity between generated answer and human answer
        similarity_score = calculate_cosine_similarity(generated_answer, human_answer, model)
        
        # Print results
        print(f"Row {index + 1}:")
        print(f"Question: {question}")
        print(f"Human Answer: {human_answer}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Cosine Similarity: {similarity_score}\n")


if __name__ == "__main__":
    csv_file = 'C:\\Users\\14107\\Desktop\\Courses\\COSC760BigData\\Project\\archive\\stage_1_qa.csv' 
    main(csv_file)
