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
import pinecone

## Load dotenv file
_ = load_dotenv(override=True)
pinecone_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')

## Connect to pinecone
pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_env
            )
index_name = 'image-vgg19model-course'
index = pinecone.Index(index_name)


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


## ------------------------------------- Search Pinecone VectorDB --------------------------------- ##
def search_vectorDB(image_url: str, folder_path: str, top_k: int, threshold: float=None):
    ''' This Function is to use the pinecone index to make a query and retrieve similar records.
    Args:
    *****
        (image_url: str) --> The image link provided to get similar records with it.
        (folder_path: str) --> The folder path in which the image will be downloaded. 
        (top_k: int) --> The number required of similar records in descending order.
        (threshold: float) --> The threshold to filter the retrieved IDs based on it.
    
    Returns:
    *******
        (similar_records: List) --> A List of dictionaries, each with id and score similartiy.
    '''
     
    try:
        ## Call the function (download_images) --> Downloading the image
        image_path_local = download_images(image_url=image_url, folder_path=folder_path)

        ## Call the function (extract_images_features) --> Feature Extraction
        image_features = extract_images_features(images_paths=[image_path_local])[0]

        ## Search pinecone
        results = index.query(vector=image_features, top_k=top_k)['matches']

        ## Filter the output if there is a threshold given
        if threshold is not None:
            ## Exatract IDs with scores
            similar_records = [{'id': int(record['id']), 'score': float(record['score'])} \
                                for record in results if float(record['score']) > threshold]       
        ## No Filtering
        else:
            ## Exatract IDs with scores
            similar_records = [{'id': int(record['id']), 'score': float(record['score'])} for record in results]
        
        return similar_records

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to get simialr records.' + str(e))
    



## ------------------------------------------ Upsert New Data to Pinecone --------------------------------------------- ##
def insert_vectorDB(image_id: int, image_url: str, folder_path: str):
    ''' This Function is to index the new images using the provided image id and link.
   
    Args:
    *****
        (image_id: int) --> The provided image id to be renamed with it.
        (image_url: str) --> The provided image link to be downloaded.
        (folder_path: str) --> The folder path to download the image on it.

    Returns:
    ********
        Only a small message
    '''
    try:
        ## Call the function (download_images) --> Dwonloading the image
        new_image_path_local = download_images(image_url=image_url, folder_path=folder_path)

        ## Call the function (extract_images_features) --> Features Extraction
        image_features = extract_images_features(images_paths=[new_image_path_local])[0]

        ## Upsert to pinecone
        to_upsert = list(zip([str(image_id)], [image_features]))

        ## Insert to pinecone
        _ = index.upsert(vectors=to_upsert)

        ## Get the count of vector after upserting
        count_after = index.describe_index_stats()['total_vector_count']

        return {f'Upserting Done: Vectors Count Now is: {count_after} ..'}

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to upsert to pinecone vector DB.')
    


## ------------------------------------ Delete Vectors form Pinecone --------------------------------------- ##
def delete_vectorDB(image_id: int):

    try:
        ## Delete from Vector DB
        _ = index.delete(ids=[str(image_id)])

        ## Get the count of vector after upserting
        count_after = index.describe_index_stats()['total_vector_count']

        return {f'Deleting Done: Vectors Count Now is: {count_after} ..'}

    except Exception as e:
        raise HTTPException(status_code=500, detail='Failed to delete from pinecone vector DB.')