## Import Libraries
from fastapi import FastAPI, HTTPException, Form
import os
from utils import search_vectorDB, index_vectorDB, delete_vectorDB  ## My Custom Functions


## Intialize an app
app = FastAPI(debug=True)


## --- First EndPoint Variables ---- ##
## Folder path for searching endpoint for the downloaded images.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DOWNLOADED_IMAGES_FOLDER_PATH = os.path.join(PROJECT_PATH, 'downloads_images')
os.makedirs(DOWNLOADED_IMAGES_FOLDER_PATH, exist_ok=True)


@app.post('/image-search')
async def image_search(image_url: str=Form(...), top_k: int=Form(...), 
                       threshold: float=Form(None), 
                       class_type: str=Form(..., description='class_type', enum=['All', 'class-a', 'class-b'])):
    
    ## Validate top_k and threshold
    if  top_k <= 0 or not isinstance(top_k, int) or top_k > 10000: ## 10000 to not get Timeouterrir (You can get any number)
        raise HTTPException(status_code=400, detail="Bad Request: 'top_k' must be between integer and between 1 and 10000.")

    elif threshold is not None and (threshold <= 0.0 or not isinstance(threshold, float) or threshold > 1.0):
        raise HTTPException(status_code=400, detail="Bad Request: 'threshold' must be between 0 and 1.")

    ## Call the Function (search_vectorDB) --> Search the Qdrant Client
    results = search_vectorDB(image_url=image_url, folder_path=DOWNLOADED_IMAGES_FOLDER_PATH, 
                              top_k=top_k, threshold=threshold, class_type=class_type)

    ## After Getting the response -- Remove the images
    _ = [os.remove(os.path.join(DOWNLOADED_IMAGES_FOLDER_PATH, image)) for image in os.listdir(DOWNLOADED_IMAGES_FOLDER_PATH)]

    return results



## --- Second EndPoint Variables ---- ##
## Folder path for indexing endpoint for the downloaded images.
INDEXING_IMAGES_FOLDER_PATH = os.path.join(PROJECT_PATH, 'indexing_downloads_images')
os.makedirs(INDEXING_IMAGES_FOLDER_PATH, exist_ok=True)


## --------------------------------- EndPoint for Indexing new Images ------------------------------ ##
@app.post('/indexing_or_deleting_images')
async def indexing_or_deleting_images(image_id: int=Form(...), image_url: str=Form(None), 
                                      class_type: str=Form(None, description='class_type', enum=['class-a', 'class-b']),
                                      case: str=Form(..., description='case', enum=['upsert', 'delete'])):   ## Handling both PERFECTLY :D

    ## Validate image_url based on the selected case   
    if case == 'upsert' and (not image_url or not class_type):
        raise HTTPException(status_code=400, detail='"image_url & class_type" are mandatory for case "upsert"')

    if case == 'upsert':
        ## Call the (index_vectorDB) from utils.py --> To index new images to the Vector Database
        message = index_vectorDB(image_id=image_id, image_url=image_url, 
                                 folder_path=INDEXING_IMAGES_FOLDER_PATH, class_type=class_type)

        ## After Getting the response -- Remove the images
        _ = [os.remove(os.path.join(INDEXING_IMAGES_FOLDER_PATH, image)) for image in os.listdir(INDEXING_IMAGES_FOLDER_PATH)]

    elif case == 'delete':
        ## Call the (delete_points_from_vectorDB) from utils.py --> To delete images with the give ids
        message = delete_vectorDB(image_id=image_id)
    
    return {'message': message}