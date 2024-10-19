#rag assessment
#step 0 -  baseline assessment of llm
#  loop through spreadsheet 
#    get question, answer, file
#    ask llama3.1 each question. 
#    find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#    save generated answer, correctness and elapsed time

#step 1 - first rag (not really a rag...)
#  loop through spreadsheet
#    get question, answer, file
#    if file != prev file then upload file , save elapsed time
#    ask question
#    get generated answer
#    find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#    save elapsed time
#

#step 2 - first chunks  <<<where we're at
#     loop through spreadsheet
#       get question, answer, file
#       if file != prev file then process file, save elapsed time
#       vectorize question
#       similarity question vector, chunks vector
#       rank similarity - top 5?
#       feed top 5? chunks and question to llm
#       find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#       save elapsed time

#step 2.5
#     everything in 2 and...
#     finds bag of words for each chunk
#     places chunks in neo4j as node
#     creates an edge from node a to node b for every word in common in their bag of words
#     find bag of words of a question and consider using chunks with similar words
#     send everything to llm to evaluate

#step 3 - questions/answer
#     loop through spreadsheet
#       get question, answer, file
#       if file != prev file then process file, save elapsed time
#           from chunk, have the llm create answers from the chunk
#                 then have the llm create questions for the generated answers
#                 then get the embedding for the generated question
#           vectorize the  generated questions 
#       get question, vector question
#       similarity question vector, chunks vector
#       rank similarity - top 5?
#       feed top 5? chunks and question to llm
#       find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#       save elapsed time

#step 4 - query decomposition, questions/answer
#     loop through spreadsheet
#       get question, answer, file
#       if file != prev file then process file, save elapsed time
#           from chunk, have the llm create questions that the chunk has answers for
#           vectorize the  generated questions 
#       get question, vector question
#       similarity question vector, chunks vector
#       rank similarity - top 5?
#       feed top 5? chunks and question to llm
#       find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#       save elapsed time
