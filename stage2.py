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
def query_ollama(question,file_content):
    ollama_api_url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        # "context": context,  # Ensure this is formatted correctly
        "prompt": f"{file_content}\n\nQuestion: {question}",
        "model": "rag3.2:1b",  # Adjust based on your Ollama model
        "stream": False
    }

    try:
        response = requests.post(ollama_api_url, json=payload, headers=headers)
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

def main(csv_file,file):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Lightweight SBERT model
    
    # Load data from csv
    data = load_data_from_csv(csv_file)
    data = data[0:2]
    last_file_name = ""
    for index, row in data.iterrows():
        print(datetime.now())
        file.write(str(datetime.now()))
        question = row['question']
        human_answer = row['human_answer']
        file_name = row['file_name']
        
        # Load the content of the file - but only if it's a different file
        if file_name != last_file_name:
            last_file_name = file_name
            file_content = load_file_content(file_name)
            #chunk file_content
            #run  sentence transformer on chunks
            #save  embeddings, save chunks
        
        #get vector ofquestion
        #findcosine similarity of  vector vs all in storage
        #sort - take top 5
        #get chunks
        #chunks + question -> llm

        
        # Query Ollama
        generated_answer = query_ollama(question,file_content)
        # generated_answer = query_ollama("", question)

        if generated_answer is None:
            print(f"Skipping row {index + 1} due to API error.")
            file.write(f"Skipping row {index + 1} due to API error.\n")
            continue  # Skip further processing for this row if no valid answer
        
        # Calculate cosine similarity between generated answer and human answer
        similarity_score = calculate_cosine_similarity(generated_answer, human_answer, model)
        
        # Print results
        print(f"Row {index + 1}:")
        print(f"Question: {question}")
        print(f"Human Answer: {human_answer}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Cosine Similarity: {similarity_score}\n")
        
        file.write(f"Row {index + 1}:\n")
        file.write(f"Question: {question}\n")
        file.write(f"Human Answer: {human_answer}\n")
        file.write(f"Generated Answer: {generated_answer}\n")
        file.write(f"Cosine Similarity: {similarity_score}\n\n")


if __name__ == "__main__":
    with open("stage2_responses.txt","+a") as file:
        csv_file = 'C:\\Users\\14107\\Desktop\\Courses\\COSC760BigData\\Project\\archive\\stage_1_qa.csv' 
        main(csv_file,file)
        file.write(str(datetime.now()))
    file.close()
