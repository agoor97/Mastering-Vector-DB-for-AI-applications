## Import Libraries
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException
import openai
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, PointIdsList

## Load dotenv file
_ = load_dotenv(override=True)
qdrant_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')

## Assign OpenAI Key
openai.api_key = os.getenv('OPENAI_API_KEY')

## Connect to Qdrant Client
client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=80) ## Increase TimeOut if the retrieving is too much

## Load the Model
model_hugging = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2', device='cpu')

## ------------------------------------ Translate using gpr-3.5-turo  --------------------------------------- ##
def translate_to_english_gpt(user_prompt: str):
    ''' This Function takes the input text and translate it to English using gpt-3.5-turbo.

        Args:
        *****
            (user_prompt: str) --> The input text that we want to translate to English.
        
        Returns:
        ********
            (translated_text: str) --> The translation of the input text to English Language.
        '''

    ## Intialize a system prompt for translating the text to English
    system_prompt = f''' You will provided with the following information.
                        1. An arbitrary input text. The text is delimited with triple backticks. 

                        Perform the following tasks:
                        1. Translate the following English text to English.
                        2. Return only the translation. Do not provide any additional information in your response.
                        3. Also, Do not require any additional information for doing your tasks.

                        Input text: ```{user_prompt}```

                        Your response:
                     '''
    ## Prepare messages
    messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
                ]


    ## Call the OPENAI Model (gpt-3.5-turbo)
    translated_text = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=messages,
                        temperature=0.7,  ## Let's give it a try.
                        max_tokens=200)  
    
    translated_text = translated_text['choices'][0]['message']['content']

    ## Some Validation
    if not translated_text:
        raise ValueError('Failed to translate the text.')


    return translated_text


## ------------------------------------- Getting Similar IDs using Qdrant ------------------------------------ ##
def search_vectDB(query_text: str, top_k: int, threshold: float=None):
    ''' This Function is to use the Qdrant index to make a query and retrieve similar records.
    Args:
    *****
        (query_text: str) --> The query text to get similar records to it.
        (top_k: int) --> The number required of similar records in descending order.
        (threshold: float) --> The threshold to filter the retrieved IDs based on it.
    
    Returns:
    *******
        (similar_ids: List) --> A List of IDs for similarity records.
    '''
    try:
        ## Call the above Function for translation for better results
        query_translated = translate_to_english_gpt(user_prompt=query_text)

        ## Get Embeddings of the input query
        query_embedding = model_hugging.encode(query_translated).tolist()

        ## Search in qdrant
        results = client.search(collection_name='semantic-search-course', query_vector=query_embedding, limit=top_k, score_threshold=threshold)

        ## Take only the id and score
        results = [{'id': point.id, 'score': float(point.score)} for point in results]

        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to get similar records' + str(e))



## ------------------------------------ Upsert New Data to Qdrant --------------------------------------- ##
def insert_vectorDB(text_id: int, text: str):
    ''' This Function is to index the new images using the provided image id and link.
   
    Args:
    *****
        (text_id: int) --> The provided text id to be inserted to Qdrant DB.
        (text: str) --> The provided text to be inserted.

    Returns:
    ********
        Only a small message with the new count of points after upserting.
    '''


    try:
        ## You can tranaslate if you want ..
        #         
        ## Get Embeddings using HuggingFace model
        embeds_new = model_hugging.encode(text).tolist()

        ## Insert to Qdrant
        _ = client.upsert(collection_name='semantic-search-course', 
                          wait=True, 
                          points=[PointStruct(id=text_id, vector=embeds_new)])

        ## Check count after Upserting
        count_after_upsert = client.get_collection(collection_name='semantic-search-course').vectors_count

        return f'Upserting Done: Count Now is {count_after_upsert} vectors.'
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to upsert to Qdrant vector DB.')



## ------------------------------------ Delete Vectors form Qdrant --------------------------------------- ##
def delete_vectorDB(text_id: int):
    ''' This Function is to delete imagesfrom the vector stores base on the provided ids.
    
    Args:
    *****
        (text_id: int) --> The provided text id to be delete.

    Returns: 
        Only a small message with the new count of points after deleting.
    '''

    try:
        ## Delete from Vector DB
        _ = client.delete(collection_name='semantic-search-course', points_selector=PointIdsList(points=[text_id]))
   
        ## Check count after Deleting
        count_after_delete = client.get_collection(collection_name='semantic-search-course').vectors_count

        return f'Deleting Done: Count Now is {count_after_delete} vectors.'
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to delete from Qdrant vector DB.')


