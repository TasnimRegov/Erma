
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import openai


load_dotenv('.envir')
openaikey = os.environ.get("openai")
pineconekey = os.environ.get("pinecone")

# EMBEDDING
model_name = 'text-embedding-ada-002'
EMBEDDING = OpenAIEmbeddings(model=model_name, api_key=openaikey)

# CALL PINECONE INDEX
pc = Pinecone(api_key=pineconekey)
index_name = "fourthindex"
index = pc.Index(index_name)


#--------------------------------------------------------------------------------------#
# RETRIEVE QUERY

# find the best match from index    
def find_k_best_match(query_text,k):
    query_em = EMBEDDING.embed_query(str(query_text))
    result = index.query(vector=query_em, top_k=k, includeMetadata=True)
    
    # return result
    return [result['matches'][i]['metadata'] for i in range(k)],[result['matches'][i]['metadata'] for i in range(k)]


    # metadata = [match['metadata'] for match in result['matches'] ]
    # ids = [match['id'] for match in result['matches'] ]
    # score =  [match['score'] for match in result['matches'] ]
    # return metadata, ids, score

    # matches = result['matches']  # or however the matches are structured in the result
    # best_match_texts = [match['metadata']['text'] for match in matches]  # Extract text from metadata
    # return "\n\n".join(best_match_texts)

    # matches = result.get('matches', [])
    # # Get the metadata or text from each match
    # best_match_texts = [match['metadata'] for match in matches if 'metadata' in match and 'text' in match['metadata']]
    # return best_match_texts



#------------------------------------------------------------------------------------#
# LLM INTEGRATION
def prompt(context, query):
    header = """
    Answer the question as truthfully as possible using the provided context, 
    and if the answer is unrelated 'Sorry Not Sufficient context to answer query' \n"
    """
    # return header + context + "\n\n" + query + "\n"
    return header + context + "\n\n" + query + "\n"   

# feed the prompt to the model to return the answer using openai's compleation api
def get_answer(promt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=promt,
        temperature=0.5,
        max_tokens=100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response.choices[0].text



#-----------------------------------------------------------------------------------#
# QUERY TEST
query = st.text_input("Enter your query")
button = st.button("Submit")

if button:
    with st.spinner("Finding an answer..."):
        title, res = find_k_best_match(query,2)
        context = "\n\n".join(res)

        st.expander("Context").write(context)
        prompt = prompt(context,query)
        answer = get_answer(prompt)
        st.success("Answer: "+ answer)



# if button:
#     with st.spinner("Finding an answer..."):
#         best_match_texts  = find_k_best_match(query)
#         if not best_match_texts:
#             st.error("No relevant context found.")
#         else:
#             context = "\n\n".join(best_match_texts)
#             st.expander("Context").write(context)
#             prompt_text = prompt(context, query)
#             answer = get_answer(prompt_text)
#             st.success("Answer: " + answer)




# if button:
#     with st.spinner("Finding an answer..."):
#         title, res = find_k_best_match(query,2)
#         context = "\n\n".join(res)

#         st.expander("Context").write(context)
#         prompt = prompt(context,query)
#         answer = get_answer(prompt)
#         st.success("Answer: "+ answer)

