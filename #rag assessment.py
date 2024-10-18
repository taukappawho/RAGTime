#rag assessment
#step 0 -  baseline assessment of llm
#  loop through spreadsheet 
#    get question, answer, file
#    ask llama3.1 each question. 
#    find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#    save generated answer, correctness and elapsed time

#step 1 - first rag
#  loop through spreadsheet
#    get question, answer, file
#    if file != prev file then upload file , save elapsed time
#    ask question
#    get generated answer
#    find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#    save elapsed time
#

#step 2 - first chunks (uses neo4j)
#     loop through spreadsheet
#       get question, answer, file
#       if file != prev file then process file, save elapsed time
#       vectorize question
#       similarity question vector, chunks vector
#       rank similarity - top 5?
#       feed top 5? chunks and question to llm
#       find cosine similarity (sentenceTransformer(generated answer), sentenceTransformer(answer)) - save result
#       save elapsed time

#step 3 - questions/answer
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