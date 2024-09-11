#https://medium.com/@Jaybrata/exploring-vector-databases-and-content-retrieval-with-pinecone-7150fb3c8ee9 

import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv('.envir')
openaikey = os.environ.get("openai")
pineconekey = os.environ.get("pinecone")


# EMBEDDING
# minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model_name = 'text-embedding-ada-002'
EMBEDDING = OpenAIEmbeddings(model=model_name, api_key=openaikey)


# LOAD DATA
df = pd.read_csv("dataset/CoffeeMenu_.csv")

payload = []
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=30,
    separators=["\n\n", "\n"]
)
idx = 1


for _,row in df.iterrows():

    text = (
        f"Category: {row['Category']}, "
        f"Beverage: {row['Beverage']}, "
        f"Size: {row['Size']}, "
        f"Milk: {row['Milk']}, "
        f"Calories: {row['Calories']}, "
        f"TotalFat: {row['TotalFat']}, "
        f"TotalCarb: {row['TotalCarb']}, "
        f"Cholesterol: {row['Cholesterol']}, "
        f"Sugars: {row['Sugars']}, "
        f"Protein: {row['Protein']}"
    )
        
    splitted_text = r_splitter.split_text(text)

    for chunk_id, s_text in enumerate(splitted_text):
        temp_dict = {}
        temp_dict["id"] = str(idx)
        temp_dict["values"] = EMBEDDING.embed_documents([s_text])[0] # generate text embedding for each text segment
        temp_dict['metadata'] = {
            'Category': row['Category'],
            'Beverage': row['Beverage'],
            'Size': row['Size'],
            'Milk': row['Milk'],
            'Calories': row['Calories'],
            'TotalFat': row['TotalFat'],
            'TotalCarb': row['TotalCarb'],
            'Cholesterol': row['Cholesterol'],
            'Sugars': row['Sugars'],
            'Protein': row['Protein'],
            'chunk_id': chunk_id,
            'text': s_text
            } #Max metadata size per vector is 40 KB
        payload.append(temp_dict)
        idx += 1



# CREATE PINECONE INDEX
pc = Pinecone(api_key=pineconekey)
index_name = "fourthindex"
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name, 
        dimension=1536, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1',
        )
)
index = pc.Index(index_name)

# # JUST UPSERT THE ID AND VALUES NOT INCLUDE THE METADATA.
# vectors =[(item['id'], item['values']) for item in payload]  # Certain Data only
# index.upsert(vectors=vectors)
# print("Data upserted successfully.")

# INGEST ALL DATA
index.upsert(vectors=payload)
print("Data upserted successfully.")

