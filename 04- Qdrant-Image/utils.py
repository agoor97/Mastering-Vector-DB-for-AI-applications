## Import Libraries
import requests
import os
from fastapi import HTTPException
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointIdsList, PointStruct, Filter, MatchValue, FieldCondition

## Load dotenv file
_ = load_dotenv(override=True)
qdrant_key = os.getenv('QDRANT_API_KEY')
qdrant_url = os.getenv('QDRANT_URL')

collec_name = 'image-search-course'
## Connect to Qdrant Client
client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=80) ## Increase TimeOut if the retrieving is too much


## Load The Moodel
model_inference = timm.create_model('vgg19', pretrained=True)
## Remove the top classifier, take only the conv base
model_inference = nn.Sequential(*list(model_inference.children())[:-1])  
_ = model_inference.eval()



## ------------------------------------- Download the image Locally --------------------------------- ##
def download_images(image_url: str, folder_path: str):
    ''' This Function is to download images only using the provided image link.
    
    Args:
    *****
        (image_url: str) --> The provided image lik to be downloaded.
        (folder_path: str) --> The folder path to download the image on it.

    Returns:
    ********
        (image_local_path: str) --> The local path of the downloaded image.
    '''
    try:
        ## Request
        response = requests.get(image_url)
        ## Continue if it is ok
        if response.status_code == 200:
            image_basename = os.path.basename(image_url).split('.')[0]

            ## Prepare the path in which you will download the image
            image_path_local = os.path.join(folder_path, f'{image_basename}.png')

            ## Wrtite the image to the local path
            with open(image_path_local, 'wb') as f:
                f.write(response.content)        
        
            return image_path_local
        
        else:
            raise HTTPException(status_code=400, detail='There is an error in downloading the image, check it again.')

    except:
        raise HTTPException(status_code=400, detail='There is an error in downloading the image, check it again.')




## ------------------------------------- Extract features maps from image --------------------------------- ##
def extract_images_features(images_paths: list):
    ''' This Function is to extract the features from the images. It can work in batches by taking a list of images paths.
    
    Args:
    *****
        (images_paths: List) --> The List that contains the images paths to extract features from them.

    Returns:
    ********
        (batch_features: List) --> The Extracted Features from the provide list of images paths.
    '''

    ## Transforming before extraction
    transform = transforms.Compose([
                        ## VGG19 requires 224x224 images
                        transforms.Resize((224, 224)), 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),  
                                  ])

    ## Looping over the images_paths
    batch_features = []
    for image_path in images_paths:
        ## Convert it to Pillow and then to tensor, and return it
        image_tensor = Image.open(image_path).convert("RGB")
        image_tensor = transform(image_tensor).unsqueeze(0)

        ## Pass the Image and get the feature extraction.
        with torch.no_grad():
            conv_features = model_inference(image_tensor)
            ## Flatten the convolutional features
            image_features = conv_features.view(conv_features.size(0), -1).tolist()[0]

        ## Append to the main list
        batch_features.append(image_features)

    return batch_features


## ------------------------------------------- Function to Search --------------------------------------------- ##
def search_vectorDB(image_url: str, folder_path: str, top_k: int, threshold: float=None, class_type: str=None):

    try:
        ## Call the function (download_images)  --> Download the image locally
        image_path_local = download_images(image_url=image_url, folder_path=folder_path)
        ## Call the function (extract_images_features) --> Extracting the features
        image_feats = extract_images_features(images_paths=[image_path_local])

        if class_type in ['class-a', 'class-b']:
            ## Search Qdrant  with filtering
            results = client.search(collection_name=collec_name, query_vector=image_feats[0], 
                                    limit=top_k, score_threshold=threshold, 
                                    query_filter=Filter(must=[FieldCondition(key='class', match=MatchValue(value=class_type))]))
        else:
            ## Search Qdrant Client
            results = client.search(collection_name=collec_name, query_vector=image_feats[0], 
                                    limit=top_k, score_threshold=threshold)

        ## Take only the id and score
        results = [{'id': int(point.id), 'score': float(point.score), 'class': point.payload['class']} for point in results]

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=f'There is a problem in searching Qdrant DataBase .. {str(e)}')



## ------------------------------------------- Function to Index new Data --------------------------------------------- ##
def index_vectorDB(image_id: int, image_url: str, folder_path: str, class_type: str):
    ''' This Function is to index the new images using the provided image id and link.
   
    Args:
    *****
        (image_id: int) --> The provided image id to be renamed with it.
        (image_url: str) --> The provided image link to be downloaded.
        (folder_path: str) --> The folder path to download the image on it.
        (class_type: str) --> The class type to be inserted in data as payload to be used later in filteration.

    Returns:
    ********
        Only a small message
    '''

    try:
        ## Call the above Function to download image
        new_image_path_local = download_images(image_url=image_url, folder_path=folder_path)

        ## Extract The Feature from the Image
        image_feats = extract_images_features(images_paths=[new_image_path_local])[0]

        ## Upsert to Qdrant Client
        client.upsert(
                    collection_name=collec_name,
                    wait=True,
                    points=[PointStruct(id=image_id, vector=image_feats, payload={'class': class_type})]
                       )    

        ## Check count after Upserting
        count_after_upsert = client.get_collection(collection_name=collec_name).vectors_count

        return f'Upserting Done: Count Now is {count_after_upsert} vector.'
    
    except:
        raise HTTPException(status_code=500, detail='There is an error in downloading the Image, Check it again.')



## ------------------------------------------- Delete Points from VectorDB --------------------------------------------- ##
def delete_vectorDB(image_id: int):
    ''' This Function is to delete imagesfrom the vector stores base on the provided ids.
    
    Args:
    *****
        (image_id: int) --> The provided image id to be delete.

    Returns: 
        Only a small message with the new count of points after deleting.
    '''

    try:
        ## Delete the ids
        msg = client.delete(collection_name=collec_name, points_selector=PointIdsList(points=[image_id]))
        if msg.status == 'completed':
            ## Check count after deleteing
            count_after_delete = client.get_collection(collection_name=collec_name).vectors_count

        return f'Deleting Done: Count Now is {count_after_delete} vector.'

    except:
        raise HTTPException(status_code=500, detail='There is an error in deleting these points, Check it again.')
