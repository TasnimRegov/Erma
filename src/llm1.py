# REFFERENCES
# https://medium.com/@Jaybrata/exploring-vector-databases-and-content-retrieval-with-pinecone-7150fb3c8ee9 
# https://github.com/kakoKong/Horoscope-Chatbot/blob/main/main.py

import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

import pinecone
from langchain_community.llms import HuggingFaceHub
from langchain.retrievers import PineconeRetriever

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from langchain.llms import HuggingFacePipeline

# from transformers import pipeline
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import RetrievalQA

class bot():
    
    load_dotenv('.envir')
    openaikey = os.environ.get("openai")
    pineconekey = os.environ.get("pinecone")
    hfkey = os.environ.get("hgface")

    # EMBEDDING
    model_emb = 'text-embedding-ada-002'
    EMBEDDING = OpenAIEmbeddings(model=model_emb, api_key=openaikey)

    # CREATE PINECONE INDEX
    pc = Pinecone(api_key=pineconekey)
    index_name = "fourthindex"
    index = pc.Index(index_name)


    model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm = HuggingFaceHub(
        repo_id = model,
        model_kwargs={
            "temperature": 0.7, 
            "top_k": 50
            },
        huggingfacehub_api_token = hfkey
    )

    from langchain import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser

    template = """
    You are coffee professional. Use following piece of context to answer the question. 
    If you don't know the answer, just say you don't know.

    Context: {context}
    Question: {question}
    Answer: 

    """

    retriever = PineconeRetriever(index=index, embd_fx=EMBEDDING)

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough() } 
    | prompt 
    | llm
    | StrOutputParser() 
    )


# Outside ChatBot() class
chatbot = bot()
input = input("Ask me anything: ")
result = chatbot.rag_chain.invoke(input)
print(result)





# # Load the Hugging Face model and tokenizer
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Create a pipeline for text generation
# hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

# # Initialize the Hugging Face pipeline in LangChain
# LLM = HuggingFacePipeline(pipeline=hf_pipeline)

# # Extract only the text from the dictionary before passing it to the LLM
# text_answer = " ".join([doc['metadata']['text'] for doc in index['matches']])

# # Create the prompt
# prompt = f"{text_answer} Using the provided information, answer the question"

# # Define a function to get a better query response
# def better_query_response(prompt):
#     better_answer = LLM(prompt)
#     return better_answer

# # Call the function
# final_answer = better_query_response(prompt=prompt)

# # Output the final answer
# print(final_answer)




# # Create the Pinecone retriever
# vectorstore = Pinecone(index, EMBEDDING.embed_query, "text")
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Adjust 'k' as needed

# # Load the Hugging Face LLM model
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# hf_pipeline = pipeline("text-generation", model=model_name)

# # Wrap the Hugging Face pipeline into a LangChain LLM
# llm = HuggingFacePipeline(pipeline=hf_pipeline)

# # Create the RAG chain
# rag_chain = RetrievalQA.from_chain_type(
#     llm, 
#     retriever=retriever, 
#     chain_type="stuff"
#     )

# # Function to interact with your chatbot
# def ask_chatbot(question):
#     response = rag_chain.run(question)
#     return response

# # Example usage
# question = "How much calories in caffe latte with soymilk for tall size?"
# response = ask_chatbot(question)
# print(response)