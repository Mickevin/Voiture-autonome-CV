import numpy as np
import imgaug as ia
from cv2 import imread, resize
import random
import imgaug.augmenters as iaa
from tensorflow.keras.utils import Sequence, to_categorical


class MY_Generator(Sequence):
    def __init__(self, image_filenames, batch_size, augmented=False, red_shape=None):
        self.image_filenames = image_filenames
        self.batch_size = batch_size 
        self.augmented = augmented
        self.red_shape = red_shape # Possibilité de redimentioner les images
        self.seq = iaa.Sequential([
            iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)    
        ])
        
    def augment_data(self, img, seg):
        # Méthode permettant de générer une augmentation de données
        aug_det = self.seq.to_deterministic() 
        image_aug = aug_det.augment_image(img)

        segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg)+1, shape=img.shape)

        segmap_aug = aug_det.augment_segmentation_maps(segmap)
        segmap_aug = segmap_aug.get_arr_int()
        return image_aug , segmap_aug
    
    
    def read(self, file_name, data):
        # Méthode permettant de lire les fichier des répertoir 'data' ou 'label'
        if data == 'label':
            if self.red_shape != None:
                return resize(imread(f'./label/{file_name}gtFine_labelIds.png', 0), self.red_shape)
            return imread(f'./label/{file_name}gtFine_labelIds.png', 0)
        elif data == 'data':
            if self.red_shape != None:
                return resize(imread(f'./data/{file_name}leftImg8bit.png', 0), self.red_shape)
            return imread(f'./data/{file_name}leftImg8bit.png',0)

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        #Fonction permettant de charger le lot de données d'entrainement
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Si l'option Data Augmentation est activée, la méthode augment_data est appliquée
        if self.augmented:
            augmented_data = np.array([self.augment_data(self.read(file_name, 'data'), self.read(file_name, 'label')) 
                                       for file_name in batch_x])

            X = np.concatenate((np.array([self.read(file_name, 'data') for file_name in batch_x]),
                                   augmented_data[:,0]),axis=0)
            
            y =  to_categorical(np.concatenate((np.array([self.read(file_name, 'label') 
                                                          for file_name in batch_x]), augmented_data[:,1]), axis=0))
            index = list(range(len(X)))
            random.shuffle(index)
            return X[index[:len(batch_x)]], y[index[:len(batch_x)]]
        
        # Sinon, le lot de données X et y sont importée des répertoir 'data' et 'label'
        else:
            X = np.array([self.read(file_name, 'data') for file_name in batch_x])
            y = to_categorical([np.array(self.read(file_name, 'label')) for file_name in batch_x])
            return X, y 