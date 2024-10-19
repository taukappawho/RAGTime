### Guide to what's here
- acquired podcast https://www.acquired.fm/ features stories of companies
- 200 podcasts were transcribed, 80 question/answer sets created, and used in Dr. Wang's rag ai class
- Kaggle link: https://www.kaggle.com/datasets/harrywang/acquired-podcast-transcripts-and-rag-evaluation

- stage0.py - evals llm - reads a question, attempts to answer the question based purely on training
- stage1.py - reads an entire text file, answers questions associated with the file, and saves the answers, certainty, and evaluation time
- stage2.py (token_counter.py) - reads a text files, chunks it by number of sentences, stores chunks in array, creates embeddings (vectors) from chunks (store in array). reads a question, creates an embedding of the question, and has faiss perform a cosine similarity search and return the top five vector indexes. use those indexes to send the chunks represented by the top vectors to the llm along with the original question. ask the llm if the generated answer answers the question. have an llm design the prompts
#### -  yet to do
-  stage 3 - takes chunks from stage2, create a bag of words (removal of stop words,stemming,TF-IDF,lowercase), place in neo4j where the node is the chunk, the bag of words an attribute, and the edges represent shared words in their bag of words. The question's bag of words can be used to find chunks that might otherwise be overlooked by cosine similarity
-  stage 4 - evaluate the question. ask an llm to decompose the question. questions that have conjunctions are often a problem for an llm
-  stage 4 - have the llm rewrite the question
-  stage 4 - create question/answer pairs for each chunk; that is, ask an llm what answers can be found in this chunk. Then ask the llm to write questions for each of the answers. create/save embeddings of the question with each chunk. The thought is that questions are more similar to question than they are to their answers.
-  the semester should be over/project turned in before we finish all of this

### 
1. download Ollama https://ollama.com/
2. download program
3. changes that will be needed:
4. 1. look at function query_ollama. the model given will have to be created. the Modelfile already exists if you want the same behavior. The nameing format is a a little inconsistent, but in stage0.py rag3.2:3b refers to llama3.2:3b and the associated Modelfile is Modelfile323brag.
   2. ollama create rag3.2:3b -f Modelfile323brag    (this assumes that Modelfile is in the same directory that the command is being issued)
5. type  ollama serve          (this will create a server on localhost)
6. that's it
   
### Ollama Modelfile
https://github.com/ollama/ollama/blob/main/docs/modelfile.md

To use this: 
1. Save it as a file (e.g. Modelfile)
2. ollama create choose-a-model-name -f <location of the file e.g. ./Modelfile>'
3. ollama run choose-a-model-name    (this is if you want to interact with the model, the programs interact w/ server)
4. Start using the model!
