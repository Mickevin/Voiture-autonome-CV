# Sauvegarde du fichier data dans l'espace de stockage Azure
from flask import Flask, render_template, request

app = Flask(__name__)

def load_img_from_azure(name):
    # Connection à l'espace de travail d'Azure
    url = f'https://ocia0932039034.blob.core.windows.net/azureml-blobstore-f8554f92-a33d-430c-a1ff-4d9a166c55fc/UI/data/{name}_leftImg8bit.png'
    X =  array(Image.open(requests.get(url, stream=True).raw))
    imsave('./static/origine.png',X)
    return X

@app.route('/', methods=['GET'])
def california_index():
    return render_template("index.html")


@app.route('/predict/', methods=['POST'])
def result():
    from keras.models import load_model

    from azureml.core import Workspace
    from azureml.core.model import Model

    from matplotlib.pyplot import imsave
    from PIL import Image
    from numpy import array
    import requests
    
    # Connection à l'espace de travail d'Azure
    ws = Workspace(subscription_id="d5bb9744-4790-446f-b7e1-591e22995cc7",
               resource_group="OpenClassrooms",
               workspace_name="OC_IA")
    model = load_model(Model.get_model_path('Model_vgg_unet', _workspace=ws))

    if request.method == 'POST':
        name_img = request.form['name_img']
        X = array([load_img_from_azure(name_img)])
        y = array([[i.argmax() for i in u] for u in model.predict(X[:])])
        imsave('./static/pred.png',y.reshape((512,1024)))
    return render_template("prediction.html", name_img=name_img)

if __name__ == '__main__':
    app.run()