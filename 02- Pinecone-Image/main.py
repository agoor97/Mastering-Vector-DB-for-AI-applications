## Import Libraries
from fastapi import FastAPI, HTTPException, Form, Depends
import os
from utils import search_vectorDB, insert_vectorDB, delete_vectorDB  ## My Custom Functions



## Intialize an app
app = FastAPI(debug=True)


## --- First EndPoint Variables ---- ##
## Folder path for searching endpoint for the downloaded images.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DOWNLOADED_IMAGES_FOLDER_PATH = os.path.join(PROJECT_PATH, 'downloads_images')
os.makedirs(DOWNLOADED_IMAGES_FOLDER_PATH, exist_ok=True)


## --------------------------------- EndPoint for Getting Similar Images ------------------------------ ##
@app.post('/image-search')
async def image_search(image_url: str=Form(...), top_k: int=Form(...), threshold: float=Form(0.5)):

    ## Validate top_k and threshold
    if top_k <= 0 or not isinstance(top_k, int) or top_k > 10000:
        raise HTTPException(status_code=400, detail="Bad Request: 'top_k' must be between integer and between 1 and 10000.")

    elif threshold <= 0.0 or not isinstance(threshold, float) or threshold > 1.0:
        raise HTTPException(status_code=400, detail="Bad Request: 'threshold' must be between 0 and 1.")

    ## Call (search_vectorDB) from utils.py --> for searching
    results = search_vectorDB(image_url=image_url, folder_path=DOWNLOADED_IMAGES_FOLDER_PATH, top_k=top_k, threshold=threshold)

    ## After Getting the response -- Remove the images
    _ = [os.remove(os.path.join(DOWNLOADED_IMAGES_FOLDER_PATH, image)) for image in os.listdir(DOWNLOADED_IMAGES_FOLDER_PATH)]

    return results




## --- Second EndPoint Variables ---- ##
## Folder path for new images that will be upserted
INDEXING_IMAGES_FOLDER_PATH = os.path.join(PROJECT_PATH, 'indexing_images')
os.makedirs(INDEXING_IMAGES_FOLDER_PATH, exist_ok=True)



## ---------------------------------------------- EndPoint for Updates ------------------------------------------- ##
@app.post('/indexing_or_deleting')
async def indexing_or_deleting(image_id: int=Form(...), image_url: str=Form(None), 
                               case: str=Form(..., description='case', enum=['upsert', 'delete'])):

    ## Validate the new_text is not None if the case=upsert
    if case == 'upsert' and not image_url:
        raise HTTPException(status_code=400, detail='"image_url" is mandatory for case "upsert".')
    
    ## For Upserting
    if case == 'upsert':
        message = insert_vectorDB(image_id=image_id, image_url=image_url, folder_path=INDEXING_IMAGES_FOLDER_PATH)     
        ## After Getting the response -- Remove the images
        _ = [os.remove(os.path.join(INDEXING_IMAGES_FOLDER_PATH, image)) for image in os.listdir(INDEXING_IMAGES_FOLDER_PATH)]

    ## For Deleting
    elif case == 'delete':
        message = delete_vectorDB(image_id=image_id)

    return {'message': message}                           

