import numpy as np
from tensorflow.keras.models import load_model
from azureml.core.model import Model
from azureml.core import Workspace, Dataset, Datastore
import json
from PIL import Image
import requests

def init():
    global model
    keras_path = Model.get_model_path(model_name = 'Model_vgg_unet')
        
    try: 
        model = load_model(keras_path)
    except:
        print("Keras model not working")

        
def run(name):
    X = np.array(load_img_from_azure(name))
    
    return X
#np.array([[i.argmax() for i in u] for u in model.predict(X[:])[0].reshape((512, 1024, 8))])


def load_img_from_azure(name):
    # Connection à l'espace de travail d'Azure
    url = f'https://ocia0932039034.blob.core.windows.net/azureml-blobstore-f8554f92-a33d-430c-a1ff-4d9a166c55fc/UI/data/{name}leftImg8bit.png'
    return np.array(Image.open(requests.get(url, stream=True).raw))