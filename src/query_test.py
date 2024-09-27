# REFFERENCES
#https://medium.com/@Jaybrata/exploring-vector-databases-and-content-retrieval-with-pinecone-7150fb3c8ee9 

import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv('.envir')
openaikey = os.environ.get("openai")
pineconekey = os.environ.get("pinecone")

# EMBEDDING
# model_name = 'text-embedding-3-small'     # The new model provide wrong answer compared to old model.
model_name = 'text-embedding-ada-002'
EMBEDDING = OpenAIEmbeddings(model=model_name, api_key=openaikey)

# CREATE PINECONE INDEX
pc = Pinecone(api_key=pineconekey)
index_name = "fourthindex"
index = pc.Index(index_name)



# -----------------------------------------------QUERY 4------------------------------------------------#
query_text = "How much calories in caffe latte with soymilk for tall size?"

# find the best match from index    
def find_k_best_match(query_text):
    query_em = EMBEDDING.embed_query(str(query_text))
    result = index.query(vector=query_em, top_k=2, includeMetadata=True)
    
    return result
    # return [result['matches'][i]['metadata'] for i in range(k)],[result['matches'][i]['metadata']['context'] for i in range(k)]


# Call the function and store the results
metadata = find_k_best_match(query_text)

# Print results
for match in metadata['matches']:
    print(f"Match ID : {match['id']}:")
    print(f"Score : {match['score']}")
    print(f"Metadata: {match['metadata']}")
    print()



# # -----------------------------------------------QUERY 3------------------------------------------------#
# query_text = "How much calories in caffe latte with soymilk for tall size?"

# # find the best match from index    
# def find_k_best_match(query_text):
#     query_em = EMBEDDING.embed_query(str(query_text))
#     result = index.query(vector=query_em, top_k=2, includeMetadata=True)
    
#     return result

# # Call the function and store the results
# metadata = find_k_best_match(query_text)

# # Print results
# for match in metadata['matches']:
#     print(f"Match ID : {match['id']}:")
#     print(f"Score : {match['score']}")
#     print(f"Metadata: {match['metadata']}")
#     print()



# # -----------------------------------------------QUERY 2------------------------------------------------#
# query_text = "How much calories in caffe latte with soymilk for tall size?"

# # find the best match from index    
# def find_k_best_match(query_text):
#     query_em = EMBEDDING.embed_query(str(query_text))
#     result = index.query(vector=query_em, top_k=2, includeMetadata=True)
     
#     metadata = [match['metadata'] for match in result['matches'] ]
#     ids = [match['id'] for match in result['matches'] ]
#     score =  [match['score'] for match in result['matches'] ]

#     return metadata, ids, score

# # Call the function and store the results
# metadata, ids, score = find_k_best_match(query_text)

# # # Print
# for i, (match, id, score) in enumerate(zip(metadata, ids, score)):
#     print(f"ID : {id}")
#     print(f"Scores : {score}")
#     print(f"Metadata : {match}")
#     print()



# -----------------------------------------------QUERY 1------------------------------------------------#
# query_text = "How much calories in caffe latte with soymilk for tall size?"
# query_vector = EMBEDDING.embed_query(query_text)

# # To get top related contents 
# related_contents = index.query(
#     vector=query_vector,
#     top_k=2,                 # define Top related answered produced
#     include_metadata=True
#     )

# for match in related_contents['matches']:
#     print()
#     print(f"ID: {match['id']}")
#     print(f"Score: {match['score']}")
#     print(f"Metadata: {match['metadata']}")
#     print()
