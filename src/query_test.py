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

# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# EMBEDDING = HuggingFaceEmbeddings(model_name=embedding_model)


# CREATE PINECONE INDEX
pc = Pinecone(api_key=pineconekey)
index_name = "fourthindex"

index = pc.Index(index_name)


# Test QUERY
query_text = "How much calories in caffe latte with soymilk for tall size?"
query_vector = EMBEDDING.embed_query(query_text)


# To get top related contents 
related_contents = index.query(
    vector=query_vector,
    top_k=2,                 # define Top related answered produced
    include_metadata=True
    )

# print
for match in related_contents['matches']:
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print(f"Metadata: {match['metadata']}")
    print()

