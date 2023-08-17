## Import Libraries
import pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException
import openai

## Load the dotenv file
_ = load_dotenv(override=True)
pinecone_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')
openai_key = os.getenv('OPENAI_API_KEY')

## Assign OpenAI key
openai.api_key = openai_key

## Connect to Pinecone Index
index_name = 'semantic-huggingface-model-course'
pinecone.init(
    api_key=pinecone_key,
    environment=pinecone_env
        )
index = pinecone.Index(index_name)


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


## ------------------------------------- Getting Similar IDs using pinecone ------------------------------------ ##
def search_vectDB(query_text: str, top_k: int, threshold: float=None, class_type: str=None):
    ''' This Function is to use the pinecone index to make a query and retrieve similar records.
    Args:
    *****
        (query_text: str) --> The query text to get similar records to it.
        (top_k: int) --> The number required of similar records in descending order.
        (threshold: float) --> The threshold to filter the retrieved IDs based on it.
        (class_type: str) --> Which class to filter using it (class-a or class-b)
    
    Returns:
    *******
        (similar_ids: List) --> A List of IDs for similarity records.
    '''
    try:
        ## Call the above Function for translation for better results
        query_translated = translate_to_english_gpt(user_prompt=query_text)

        ## Get Embeddings of the input query
        query_embedding = model_hugging.encode(query_translated).tolist()

        if class_type in ['class-a', 'class-b']:
            ## Search in pinecone with filtering using class_type
            results = index.query(queries=[query_embedding], top_k=top_k, 
                                  filter={'class': class_type}, namespace='semantic-huggingface', include_metadata=True)
            results = results['results'][0]['matches']
        else: 
            ## Search in pinecone without filtering
            results = index.query(queries=[query_embedding], top_k=top_k, namespace='semantic-huggingface', include_metadata=True)
            results = results['results'][0]['matches']

        
        ## Filter the output if there is a threshold given
        if threshold: 
            ## Exatract IDs with scores
            similar_records = [{'id': int(record['id']), 'score': float(record['score']), 'class': record['metadata']['class']} \
                               for record in results if float(record['score']) > threshold]
       
        ## No Filtering
        else:
            ## Exatract IDs with scores
            similar_records = [{'id': int(record['id']), 'score': float(record['score']), 'class': record['metadata']['class']} for record in results]

        return similar_records
    
    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to get similar records' + str(e))



## ------------------------------------ Upsert New Data to Pinecone --------------------------------------- ##
def insert_vectorDB(text_id: int, text: str, class_type: str):

    try:
        ## You can tranaslate if you want ..
        #         
        ## Get Embeddings using HuggingFace model
        embeds_new = model_hugging.encode(text).tolist()

        ## Prepare to pinecone 
        to_upsert = [(str(text_id), embeds_new, {'class': class_type})]

        ## Insert to pinecone
        _ = index.upsert(vectors=to_upsert, namespace='semantic-huggingface')

        ## Get the count of vector after upserting
        count_after = index.describe_index_stats()['total_vector_count']

        return {f'Upserting Done: Vectors Count Now is: {count_after} ..'}

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to upsert to pinecone vector DB.')



## ------------------------------------ Delete Vectors form Pinecone --------------------------------------- ##
def delete_vectorDB(text_id: int):

    try:
        ## Delete from Vector DB
        _ = index.delete(ids=[str(text_id)], namespace='semantic-huggingface')

        ## Get the count of vector after upserting
        count_after = index.describe_index_stats()['total_vector_count']

        return {f'Deleting Done: Vectors Count Now is: {count_after} ..'}

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to delete from pinecone vector DB.')


