##########################################################################
## Import modules                                                       ##
##########################################################################
import os
import torch
from math import floor
from PIL import Image
import numpy as np
import pandas as pd
import json
import torchvision.models as models
#Vector generation
from img2vec_pytorch import Img2Vec
import cv2
#Image similarity
import math
from numpy.linalg import norm

##########################################################################
## VECTOR GENERATION                                                    ##
##########################################################################

#Returns all images (RGB) and their paths
def load_images_paths(directory):
    images = []
    paths=[]
    car_actual,archivos=os.walk(directory)
    for archivo in car_actual[2]:
        name=str(archivo)
        l = len(name)
        #filename=''.join(['0'for x in range(12-l)])+name+".jpg"
        filename=name
        image=Image.open(directory+filename)
        image.load()

        # replace alpha channel with white color
        self_im = Image.new('RGB', image.size, (255, 255, 255))
        self_im.paste(image, None)
        images.append(self_im)
        images[-1].load()
        paths.append(filename)
    return images, paths

#Vectorize all images and save it in a file .pickle
def vectorization(images,model_name = 'resnet-18'):
    print(f'Starting {model_name}')
    img2vec = Img2Vec(cuda=True, model = model_name)
    vectors=[]
    n=len(images)
    for i in range(n):
        v = img2vec.get_vec(images[i])
        vectors.append(v)
    print(f'Saving {model_name}')
    np.save(f'./embeddings/train_img_vectors_{model_name}_analysis.npy',vectors)

#Save all images with vectors in a dataframe
def save_df_vectors(models,ids,vectors):
    for i in range(len(models)):
        df= pd.DataFrame(zip(ids,vectors[i]),columns=['Image ID','Image Vector'])
        df.to_pickle(f'../../project/image_recuperation/embeddings/df_image_vectors_train_{models[i]}.pickle')

##########################################################################
## IMAGE SIMILARITY                                                     ##
##########################################################################
def euclidean_distance(x,y):
    return 1/(math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y))))

def manhattan_distance(x,y):
    return 1/(sum(abs(a-b) for a,b in zip(x,y)))

def cosine_similarity(x,y):
    return (np.dot(x,y)/(norm(x)*norm(y)))

def vector_sim(v1, v2, sim_metric):
    comp = 0
    if sim_metric == 'euc':
        comp = euclidean_distance(v1, v2)
    elif sim_metric == 'man':
        comp = manhattan_distance(v1, v2)
    elif sim_metric == 'cos':
        comp = cosine_similarity(v1, v2)
    return comp
        