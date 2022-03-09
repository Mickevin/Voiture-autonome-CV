import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os

count = collections.Counter
path_gtFine_trainvaltest = './P8_Cityscapes_gtFine_trainvaltest/gtFine/'
path_leftImg8bit_trainvaltes = './P8_Cityscapes_leftImg8bit_trainvaltest/leftImg8bit/'

def return_points(data):
    points = []
    for objects in data['objects']:
        for polygon in objects['polygon']:
            points.append([objects['label'], polygon[0], polygon[1]])
    return pd.DataFrame(points, columns=['label', 'x', 'y'])


def show_img():
    dir_path = path_leftImg8bit_trainvaltes + 'train/aachen/'
    dir_path_ = path_gtFine_trainvaltest + 'train/aachen/'
    list_dir = os.listdir(dir_path)
    file_id = list_dir[np.random.randint(len(list_dir))].split('leftImg8bit.png')[0]
    
    
    im0 = cv2.imread(dir_path+file_id+'leftImg8bit.png', cv2.IMREAD_UNCHANGED)
    im1 = cv2.imread(dir_path_+file_id+'gtFine_color.png', cv2.IMREAD_UNCHANGED)
    im2 = cv2.imread(dir_path_+file_id+'gtFine_instanceIds.png',cv2.IMREAD_UNCHANGED)
    im3 = cv2.imread(dir_path_+file_id+'gtFine_labelIds.png', cv2.IMREAD_UNCHANGED)
    
    
    plt.figure(figsize=(20,12))

    plt.subplot(2,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Image d'origine 'leftImg8bit.png', taille de l'image : {im0.shape}")
    plt.imshow(im0)
    plt.subplot(2,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Image du fichier 'gtFine_color.png', taille de l'image : {im1.shape}")
    plt.imshow(im1)
    plt.subplot(2,2,3)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Image du fichier 'gtFine_instanceIds.png', , taille de l'image : {im2.shape}")
    plt.imshow(im2)
    plt.xlabel(f'N label : {len(set(im2.flatten()))}')
    plt.subplot(2,2,4)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Image du fichier 'gtFine_labelIds.png, , taille de l'image : {im3.shape}'")
    plt.imshow(im3)
    plt.xlabel(f'N label : {len(set(im3.flatten()))}')
    plt.show()
    
    with open(dir_path_+file_id+'gtFine_polygons.json') as json_file:
        data = json.load(json_file)

    points = []
    for objects in data['objects']:
        for polygon in objects['polygon']:
            points.append([objects['label'], polygon[0], polygon[1]])
    
    points = pd.DataFrame(points, columns=['label', 'x', 'y'])
    
    plt.figure(figsize=(20,15))
    plt.title("Image avec localisation des labels provenant du fichier .json")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(im3)
    sns.scatterplot(data=points, x="x", y="y", hue='label')
    plt.show()
    return data, im0, im1, im2, im3

def show_2img(img, img_):
    plt.figure(figsize=(20,12))
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Image d'origine de taille de l'image : {img.shape}")
    plt.imshow(img)
    
    plt.subplot(1,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Image transformée de taille de l'image : {img.shape}")
    plt.imshow(img_)
    plt.show()
    
def evaluate_loss(loss, y_true, y_pred):
    val = np.array([loss(to_categorical(y_true[n], 8).astype('float32'),
                         to_categorical(y_pred[n], 8).astype('float32')).numpy() 
                    for n in range(len(y_pred))])
    
    print(f"Loss max : {val[i]}, loss min : {val[u]}, loss mean :{val.mean()}")
    plt.figure(figsize=(15,6))
    sns.violinplot(x=val)
    plt.show()

    i = val.argmax()
    show_2img(y_true[i], y_pred[i])

    u = val.argmin()
    show_2img(y_true[u], y_pred[u])
    
def plt_learningcurve(model, metric):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    metrics = model.history[metric]
    val_metric = model.history[f'{metric}']

    plt.figure(figsize=(20,9))
    plt.subplot(1,2,1)
    plt.plot(model.epoch, metrics)
    plt.plot(model.epoch, val_metric)
    plt.legend([metric,f'{metric}'])
    
    plt.subplot(1,2,2)
    plt.plot(model.epoch, loss)
    plt.plot(model.epoch, val_loss)
    plt.legend(['Loss','Validation_Loss'])
    

def evaluate_loss(loss, y_true, y_pred):
    val = np.array([loss(to_categorical(y_true[n], 8).astype('float32'),
                         to_categorical(y_pred[n], 8).astype('float32')).numpy() 
                    for n in range(len(y_pred))])
    
    
    plt.figure(figsize=(15,6))
    sns.violinplot(x=val)
    plt.show()

    i = val.argmax()
    show_2img(y_true[i], y_pred[i])

    u = val.argmin()
    show_2img(y_true[u], y_pred[u])
    print(f"Loss max : {val[i]}, loss min : {val[u]}, loss mean :{val.mean()}")
    
    
    
def plt_learningcurve(model, metric):
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    metrics = model.history[f'val_{metric}']
    val_metric = model.history[f'{metric}']

    plt.figure(figsize=(20,9))
    plt.subplot(1,2,1)
    plt.plot(model.epoch, metrics)
    plt.plot(model.epoch, val_metric)
    plt.legend([metric,f'{metric}'])
    
    plt.subplot(1,2,2)
    plt.plot(model.epoch, loss)
    plt.plot(model.epoch, val_loss)
    plt.legend(['Loss','Validation_Loss'])