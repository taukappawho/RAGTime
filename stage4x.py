import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import  BertTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime
import numpy as np
import faiss
import re
# #nltk stuff
import nltk
from nltk.corpus import stopwords
import string
nltk.download("stopwords")
#neo4j stuff
from neo4j import GraphDatabase

MAX_TOKEN_COUNT = 512
tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-minilm-l12-v2')
model = SentenceTransformer('sentence-transformers/all-minilm-l12-v2')

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
def query_ollama(question, model):
    ollama_api_url = "http://localhost:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
    
    payload = {
        "prompt": f"{question}",
        # "model": "rag3.2:3b",  # Adjust based on Ollama model
        "model": f"{model}",  # Adjust based on Ollama model
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
def chunks(sentences, start, num, prev):
    if start == 0:
        prev = 0
    last = start + num
    if last >  len(sentences):
        last = len(sentences) 
    chunk = ". ".join(sentences[start-prev: last])
    return chunk

#takes text, returns a list of chunks and a list of embeddings. chunk size set by prev and num
def text_chunk(text):
    sentences = re.split(r'[.?!]', text)
    start = 0   #index to the first sentence of new chunk
    num = 8    #number of new sentences to addd to chunk
    prev = 3    #number of previous sentences to add to chunk
    index = 0   #chunk #
    embeddings = []
    chunk_list = []
    exceed = 0
    while start < len(sentences):
        chunk = chunks(sentences,start,num,prev)
        embedding = model.encode(chunk)
        num_words = len(chunk.split(" "))
        if num_words > 256:
            exceed = exceed + 1       
        embeddings.append(embedding)
        chunk_list.append(chunk)
        start = start + num
        index = index + 1
    print(f"Chunk total: {index}\nword count exceeded: {exceed}")
    return [chunk_list, embeddings]

def out(string, file):
    print(string)
    file.write(string+'\n')
    
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return words
   
def get_top_indices(nums, n=5):
    sorted_indices = sorted(enumerate(nums), key=lambda x: x[1], reverse=True)
    # Extract the top `n` indices
    top_indices = [index for index, value in sorted_indices[:n]]
    return top_indices
    
# Initialize Neo4j driver 
uri = "bolt://localhost:7687"
username = "neo4j"
password = "password"
driver = GraphDatabase.driver(uri, auth=(username, password))

# Function to create nodes and relationships in Neo4j
def create_chunks_in_neo4j(chunk_list, words):
    with driver.session() as session:
        # Step 1: Create nodes for each chunk
        for idx, chunk in enumerate(chunk_list):
            session.run(
                """
                MERGE (c:Chunk {id: $id, text: $text})
                """,
                id=idx, text=chunk
            )
        
        # Step 2: Create relationships based on shared words
        for word, indices in words.items():
            if indices:  # Skip if `indices` is empty
                # For each pair of indices, create a relationship between chunks
                for i in range(len(indices) - 1):
                    for j in range(i + 1, len(indices)):
                        session.run(
                            """
                            MATCH (c1:Chunk {id: $id1}), (c2:Chunk {id: $id2})
                            MERGE (c1)-[:SHARES_WORD {word: $word}]->(c2)
                            """,
                            id1=indices[i], id2=indices[j], word=word
                        )

# def distinct(list):     #couldn't just convert to a set and find the size?   
#     last = list[0]
#     distinct = 0
#     for ele in list:
#         if ele != last:
#             last = ele
#             distinct = distinct + 1
#     return distinct

def add_useful(list, percent, big_list):
    ret_value = []
    arr = [0] *  len(big_list)
    num_words = len(list)
    bar = percent * num_words
    for l1 in list:
        for l2 in l1:
            arr[l2]  = arr[l2] + 1
    # print(f"arr={arr}")
    index = 0
    for idx in arr:
        # print(f"\tidx={idx}, bar={bar}")
        if idx >= bar:
            ret_value.append(index)
        index = index + 1
    # print(f"ret_val={ret_value}")
    for l1 in list:
        if len(l1) < 4:
            ret_value.extend(l1)
    return ret_value

def add_top_indices(list, percent, big_list, num):
    ret_value = []
    arr = [0] *  len(big_list)
    num_words = len(list)
    bar = percent * num_words
    for l1 in list:
        for l2 in l1:
            arr[l2]  = arr[l2] + 1
    return get_top_indices(arr, num)
    
def main(csv_file,file, p):
    
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # Lightweight SBERT model
    # Load data from csv
    data = load_data_from_csv(csv_file)
    last_file_name = ""
    bow_list = []
    words = {}
    for index, row in data.iterrows():
        out(str(datetime.now()), file)
        # file.write(str(datetime.now())+"\n")
        question = row['question']
        human_answer = row['human_answer']
        file_name = row['file_name']
        
        # Load the content of the file - but only if it's a different file
        if file_name != last_file_name:
            out(f"reading {file_name}", file)
            last_file_name = file_name
            bow_list = []
            words = {}
            file_content = load_file_content(file_name)
            #chunking done here
            [chunk_list, embeddings] = text_chunk(file_content)
            idx = 0
            for chunk in chunk_list:
                bow = preprocess_text(chunk)
                for word in bow:
                    if word not in words:
                        words[word] = []
                    if idx not in words[word]:
                        words[word].append(idx) 
                bow_list.append(bow)
                idx = idx + 1
            out("len(BOW) = " + str(len(list(words.keys()))), file)

# create graph database
#   nodes are chunks
#   edges are links between bag of words

            embedding_matrix = np.array(embeddings)
            shape = embedding_matrix.shape[1]
            index_faiss = faiss.IndexFlatL2(shape)
            index_faiss.add(embedding_matrix)
            
        query_embedding = model.encode(question)  # Query embedding (1D array)
        query_embedding = np.array([query_embedding])  # Convert to 2D array with shape (1, embedding_dimension)
        q = []
        nq = preprocess_text(question)
        out(f"Question: {question}, nq: {nq}", file)
        for word in nq:
            if word in words:
                # print(f"word: {word}")
                q.append(words[word])
        out(f"q={q}", file)  

        # Perform the search
        k = 5  # Number of nearest neighbors to retrieve
        distances, indices = index_faiss.search(query_embedding, k)  # Search the index   
        # Retrieve the corresponding chunks
        
        # ans = []
        nodes = indices[0].tolist() #the semantic matches
        out(f"semantic nodes: {nodes}", file)

            
#add useful params, 1: query, 2: percentage of query BOW in chunk to be included; 3: list of all chunks
        nodes.extend(add_useful(q, p, chunk_list))
        nodes = list(set(nodes))
        out(f"nodes prior to bespoke: {nodes}", file)
        #eval each node for final acceptance
        #stage4 changes
        nodes_accepted = []
        for node in nodes:
            document = f"document: {chunk_list[node]}"
            claim = f"claim: {question}"
            prompt = f"{document}\n{claim}"
            answer = query_ollama(prompt, "bespoke-minicheck")
            # print(f"bespoke answer: {answer}")
            if answer != "No":
                nodes_accepted.append(node)
                out(f"node[{node}] accepted", file)
        
        if len(nodes_accepted) != 0:
            nodes = nodes_accepted
        else:
            k = 5  # Number of nearest neighbors to retrieve
            distances, indices = index_faiss.search(query_embedding, k)  # Search the index   
            nodes = indices[0].tolist() #the semantic matches
            nodes.extend(add_top_indices(q,p,chunk_list,4))
        out(f"nodes after bespoke: {nodes}", file)
        out(f"#chunks: {len(chunk_list)}", file)
        i = 0
        prompt = f"Here are {len(nodes)} chunks of information relevant to the following question. Please read each chunk carefully and provide a concise answer based on the information."
        for idx in nodes[:7]:
            prompt = prompt + "\n\n" + f"Chunk {i}:\n[{chunk_list[idx]}]"
            i = i + 1
        prompt = prompt + "\n\n" + f"Question: [{question}]"
        # print(f"promtp: {prompt}")
        
        #models
        #"llama3.1:8brag"
        #"rag3.2:3b"
        #"bespoke-minicheck"
        generated_answer = query_ollama(prompt, "rag3.2:3b")
        # for idx in nodes:
        #     print(f"{idx}:\t{chunk_list[idx]}")
        # generated_answer = None
        if generated_answer is None:
            out(f"Skipping row {index} due to API error.", file)
            continue  # Skip further processing for this row if no valid answer
        
        # Calculate cosine similarity between generated answer and human answer
        similarity_score = calculate_cosine_similarity(generated_answer, human_answer, model)

        # Print results
        out(f"Row {index}:", file)
        out(f"Question: {question}", file)
        out(f"Human Answer: {human_answer}", file)
        out(f"Generated Answer: {generated_answer}", file)
        out(f"Cosine Similarity: {similarity_score}\n", file)
        
if __name__ == "__main__":
    percent = [1]
    for p in percent:
        p1 = p*10
        with open(f"stage4_{p1}_responses.txt","a+", encoding="utf-8") as file:
            out(f"stage4/llama3.2:3b/bespoke/p={p}", file)
            csv_file = '.\\archive\\stage_0_qa.csv' 
            main(csv_file,file, p)
            file.write(str(datetime.now()))
        file.close()