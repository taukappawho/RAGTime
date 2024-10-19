import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import  BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime
import numpy as np
import faiss

# tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-minilm-l12-v2')
model = SentenceTransformer('sentence-transformers/all-minilm-l6-v2')

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
    file_dir = '.\\archive\\acquired-individual-transcripts\\acquired-individual-transcripts\\'
    with open(file_dir + file_name + ".txt", 'r', encoding='utf-8', errors='replace') as file:
        return file.read()

# Step 3: Send context and question to Ollama
def query_ollama(question):
    ollama_api_url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "prompt": f"{question}",
        "model": "rag3.2:3b",  # Adjust based on your Ollama model
        "stream": False
    }

    try:
        response = requests.post(ollama_api_url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("response") 
        else:
            print(f"Error querying Ollama: {response.status_code} - {response.text}")
            return None  # Return None if there's an error
    except Exception as e:
        print(f"Exception occurred while querying Ollama: {e}")
        return None  # Return None in case of an exception

# Step 4: Generate embeddings and calculate cosine similarity of answers
def calculate_cosine_similarity(text1, text2, model):
    embeddings1 = model.encode([text1], convert_to_tensor=True)
    embeddings2 = model.encode([text2], convert_to_tensor=True)
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

#just counts tokens and returns the count, no mutations - used to determine if #senetences need to be smaller
def token_count(text):
    tokens = model.tokenizer.tokenize(text)
    count = len(tokens)
    return count

#takes some text, some parameters, and returns chunks 1 at a time
def chunks(sentences, start, num, prev,index):
    last = start + num
    if last >  len(sentences):
        last = len(sentences)
    chunk = " ".join(sentences[start-prev: last])
    tc = token_count(chunk)
    # print(f"Chunk #{index}: {tc} tokens")
    if tc > 384:
        print(f"Chunk {index} exceeded 384 tokens. {tc}")
    return chunk

#takes text, returns a list of chunks and a list of embeddings. chunk size set by prev and num
def text_chunk(text):
    sentences = text.split(".")
    for sentence in sentences:
        sentence = sentence + "."
#      model = SentenceTransformer('sentence-transformers/all-minilm-l6-v2')
    start = 0   #index to the first sentence of new chunk
    num = 5     #number of new sentences to addd to chunk
    prev = 2    #number of previous sentences to add to chunk
    index = 1   #chunk #
    count = 0   #token counter
    embeddings = []
    chunk_list = []
    # tstart = datetime.now()
    # print(tstart)
    while start < len(sentences):
        chunk = chunks(sentences,start,num,prev,index)
        # if len(chunk) > 384:
        #     count = count + 1
        embeddings.append(model.encode(chunk))
        chunk_list.append(chunk)
        start = start + num
        prev = 0
        index = index + 1
    print(f"Chunk total: {index}\nChunks with more than 384 tokens: {count}")
    index = 0
    # print(f"start: {tstart}")
    # finish = datetime.now()
    # print(f'finish: {finish}')
    # print(f"elapsed time: {finish-tstart}")
    return [chunk_list, embeddings]

def main(csv_file,file):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Lightweight SBERT model
    
    # Load data from csv
    data = load_data_from_csv(csv_file)
    last_file_name = ""
    for index, row in data.iterrows():
        print(datetime.now())
        file.write(str(datetime.now()))
        question = row['question']
        human_answer = row['human_answer']
        file_name = row['file_name']
        
        # Load the content of the file - but only if it's a different file
        if file_name != last_file_name:
            print(f"reading {file_name}")
            last_file_name = file_name
            file_content = load_file_content(file_name)
            #chunking done here
            [chunk_list, embeddings] = text_chunk(file_content)
            embedding_matrix = np.array(embeddings)
            shape = embedding_matrix.shape[1]
            index_faiss = faiss.IndexFlatL2(shape)
            index_faiss.add(embedding_matrix)
            
        query_embedding = model.encode(question)  # Query embedding (1D array)
        query_embedding = np.array([query_embedding])  # Convert to 2D array with shape (1, embedding_dimension)    
        # Perform the search
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = index_faiss.search(query_embedding, k)  # Search the index   
        # Retrieve the corresponding chunks
        
        ans = []
        for i, idx in enumerate(indices[0]):
            # print(f"Chunk {i+1} (distance: {distances[0][i]}): {chunk_list[idx]}")
            ans.append(chunk_list[idx])
        prompt = f"""
Here are 5 chunks of information relevant to the following question. Please read each chunk carefully and provide a concise answer based on all the information.

Chunk 1: 
[{ans[0]}]

Chunk 2: 
[{ans[1]}]

Chunk 3: 
[{ans[2]}]

Chunk 4: 
[{ans[3]}]

Chunk 5: 
[{ans[4]}]

Question: [{question}]

Please provide a concise and accurate answer based on the information from all the chunks.
"""
        
        # Query Ollama
        generated_answer = query_ollama(prompt)
        # generated_answer = query_ollama("", question)

        if generated_answer is None:
            print(f"Skipping row {index + 1} due to API error.")
            file.write(f"Skipping row {index + 1} due to API error.\n")
            continue  # Skip further processing for this row if no valid answer
        
        # Calculate cosine similarity between generated answer and human answer
        similarity_score = calculate_cosine_similarity(generated_answer, human_answer, model)
        judge_prompt = f"""
Question: {question}
Answer: {generated_answer}

Has the question been answered correctly? Respond with "Yes" or "No".
"""
        # Print results
        print(f"Row {index + 1}:")
        print(f"Question: {question}")
        print(f"Human Answer: {human_answer}")
        print(f"Generated Answer: {generated_answer}")
        print(f"Cosine Similarity: {similarity_score}")
        judge = query_ollama(judge_prompt)
        print(f"Has the question has been answered correctly: {judge}\n") 
            
        file.write(f"Row {index + 1}:\n")
        file.write(f"Question: {question}\n")
        file.write(f"Human Answer: {human_answer}\n")
        file.write(f"Generated Answer: {generated_answer}\n")
        file.write(f"Cosine Similarity: {similarity_score}\n\n")
        file.write(f"Has the question has been answered correctly: {judge}\n")
        
if __name__ == "__main__":
    with open("stage2_responses.txt","+a", encoding="utf-8") as file:
        csv_file = '.\\archive\\stage_0_qa.csv' 
        main(csv_file,file)
        file.write(str(datetime.now()))
    file.close()