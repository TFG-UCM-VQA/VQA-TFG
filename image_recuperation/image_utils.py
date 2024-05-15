import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from img2vec_pytorch import Img2Vec

import sys
import os
from pathlib import Path

current_dir = Path.cwd()
parent_dir = os.path.abspath(os.path.dirname(current_dir))
sys.path.append(os.path.join(parent_dir, 'image_recuperation'))

from image_tag_utils import *
from image_pix_utils import *
from image_emb_utils import * 

COMMON_SHAPE = (640, 480)

def get_images_files(dataset_name):
    return [f for f in os.listdir(f'../dataset/images/{dataset_name}') if not f.startswith('.')]

def np_image(img, b_resize = False):
    img1= Image.new('RGB', img.size, (255, 255, 255))
    img1.paste(img, None)
    if b_resize:
        img1 = img1.resize(COMMON_SHAPE)
    img1 = np.array(img1)
    return img1


def calculate_similarity(image, kind, sim_func, coco, dataset, options = '', sort = True):
    imgsIds = [int(x[:-4]) for x in get_images_files(dataset)]
    
    if kind == 'tags':
        options = options.split('|')
        precision = float(options[1])
        h = float(options[2])
        image = np_image(image)
        annIds = coco.getAnnIds(imgIds=imgsIds, iscrowd=None)
        imgsAnns = coco.loadAnns(annIds)
        if options[0] == 'RNet':
            model, device = init_RNet_model()
            boxes, classes = predict_RNet(image, model, device, precision)
        elif options[0] == 'YOLO':
            net = init_YOLO_model()
            boxes, classes = predict_YOLO(image, net, precision)
            
    elif kind == 'embeddings':
        options = options.split('|')
        img2vec = Img2Vec(model=options[0])
        vectors = np.load(f'../../project/image_recuperation/embeddings/train_img_vectors_{options[0]}_analysis.npy')
        df_image_vectors_train=pd.read_pickle(f'../../project/image_recuperation/embeddings/df_image_vectors_train_{options[0]}.pickle')
        vector_1 =img2vec.get_vec(image)
        
    elif kind == 'pixels':
        image = np_image(image, b_resize = True)
    
    counter = 0 #counter for debugging
    similarities = []
    for image_id in imgsIds:
        if kind == 'tags':
            box, tags = get_ann_tags(image_id, imgsAnns)
            if sim_func == 'set':
                similarities.append(set_similarity(classes, tags))
            elif sim_func == 'set_s':
                similarities.append(set_similarity_s(classes, tags, coco, h))
            elif sim_func == 'multiset':
                similarities.append(multiset_similarity(classes, tags))
            elif sim_func == 'multiset_s':
                similarities.append(multiset_similarity_s(classes, tags, coco, h))
            elif sim_func == 'size_set':
                similarities.append(size_set_similarity(classes, tags, box_sizes(boxes), box_sizes(box)))
            elif sim_func == 'size_set_s':
                similarities.append(size_set_similarity_s(classes, tags, box_sizes(boxes), box_sizes(box), coco, h))
                
        elif kind == 'embeddings':
            vector_2=df_image_vectors_train.loc[df_image_vectors_train['Image ID']==str(image_id),'Image Vector'].iloc[0]
            if sim_func == 'euc':
                similarities.append(vector_sim(vector_1, vector_2, 'euc'))
            elif sim_func == 'man':
                similarities.append(vector_sim(vector_1, vector_2, 'man'))
            elif sim_func == 'sqr':
                similarities.append(vector_sim(vector_1, vector_2, 'sqr'))
            elif sim_func == 'cos':
                similarities.append(vector_sim(vector_1, vector_2, 'cos'))
            elif sim_func == 'jac':
                similarities.append(vector_sim(vector_1, vector_2, 'jac'))

        elif kind == 'pixels':
            path = f'../dataset/images/{dataset}/{image_id}.jpg'
            img2 = Image.open(path)
            img2 = np_image(img2, b_resize = True)
            if sim_func == 'nrmse':
                similarities.append(pixel_sim(image, img2, 'nrmse'))
            elif sim_func == 'mse':
                similarities.append(pixel_sim(image, img2, 'mse'))
            elif sim_func == 'hd':
                counter += 1
                similarities.append(pixel_sim(image, img2, 'hd'))
                print(counter)
            elif sim_func == 'nmi':
                similarities.append(pixel_sim(image, img2, 'nmi'))
            elif sim_func == 'structural':
                a = 0

    scores = pd.DataFrame()
    scores['ids'] = imgsIds
    scores['sim'] = similarities
    if sort:
        scores = scores.sort_values(by ='sim', ascending=False)
    return scores


def top_similar(image, comp, dataset, ascending = False, top = 5):
    top_images = comp[:top]
    text = 'similarity'
    if ascending:
        top_images = comp[-top:]
        text = 'disimilarity'
        
    print('Image to compare:')
    plt.imshow(image)
    plt.axis('off') 
    plt.show()    
    
    print('Similar images:')
    max_images_per_row = 4
    num_cols = min(top, max_images_per_row)
    num_rows = math.ceil(top / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Adjust figsize as needed
    for i, ax in enumerate(axes.flat):
        if i < top:
            image_path = f"../dataset/images/{dataset}/{int(top_images.iloc[i]['ids'])}.jpg"  
            image = Image.open(image_path)
            ax.imshow(image)
            ax.axis('off')  # Turn off axis labels
            ax.set_title(f'Image {text}: {top_images.iloc[i]["sim"]}', fontsize=10, pad=5)
        else:
            ax.axis('off')  # Turn off empty subplots
    plt.tight_layout(pad = 1)
    plt.show()