##########################################################################
## Import modules                                                       ##
##########################################################################
import os
import torch
from math import floor
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torchvision.models as models
from PIL import Image
from glob import glob
import numpy as np
# import matplotlib.pyplot as plt 
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
# import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
import json
import tensorflow as tf
import torchvision.models as models
from tensorflow.keras.models import model_from_json, Model,Sequential
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import re
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, Masking, TimeDistributed
from tensorflow. keras.utils import plot_model
#Vectorizing questions
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.cluster import DBSCAN

##########################################################################
## GENERAL                                                              ##
##########################################################################

def get_questions_of_images(id_list):
    question_file = open('../dataset/annotations/v2_OpenEnded_mscoco_train2014_questions.json')
    questions_all = json.load(question_file)
    questions_all = pd.DataFrame(questions_all['questions'])
    filtered_df = questions_all[questions_all['image_id'].isin(id_list)]
    return filtered_df

##########################################################################
## SIMILARITY BETWEEN QUESTIONS                                         ##
##########################################################################

def by_image_sim(questions, images):
    df = questions.join(images.set_index('ids'), on='image_id').sort_values(by='sim', ascending=False)
    df['question_score'] = df['sim']
    return df

def qtype_sim(questions, images):
    ann_file = open('../dataset/annotations/v2_mscoco_train2014_annotations.json')
    ann_all = json.load(ann_file) 
    ann_all = ann_all['annotations']
    df = pd.DataFrame(ann_all)
    question_types = df.question_type.unique()
    df = df[['question_id', 'question_type']]
    q = questions.join(images.set_index('ids'), on='image_id')
    q = q.join(df.set_index('question_id'), on='question_id')
    typecount = q[['question_type', 'question_id']].groupby(by='question_type').count().rename(columns={'question_id':'N_type'})
    q = q.join(typecount, on='question_type')
    q['question_score'] = q['sim']*q['N_type']/max(list(q['N_type']))
    return q.sort_values(by='question_score', ascending=False)

def lista_vectores(df_vectors_array):
    l=[]
    for tf in df_vectors_array:
        l.append(np.array(tf)[0])
    return l

def calcular_centroide(tensors):
    centroid = np.mean(tensors, axis=0)
    return centroid

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum((vector1 - vector2)**2))


##########################################################################
## QUESTION RECUPERATION                                                ##
##########################################################################

def questions_recuperation(df, df_scores, num_images_rec, num_questions_rec):
    dbscan = DBSCAN(eps=0.45, min_samples=5)
    
    df_filtered = df[df['image_id'].isin(df_scores['ids'][:num_images_rec])]
    X = np.array(df_filtered['vectors'])
    df_new = pd.DataFrame([i for i in X], columns=[i for i in range(len(X[0]))])
    clusters = dbscan.fit_predict(df_new)
    df_filtered['cluster'] = clusters
    df_cq = pd.DataFrame()
    clust_ids = df_filtered[df_filtered['cluster'] != -1]['cluster'].unique().tolist()
    df_cq['clusterId'] = clust_ids
    size = []
    questions = []
    for i in clust_ids:
        r = df_filtered[df_filtered['cluster'] == i]
        size.append(len(r))
        centroide = calcular_centroide(r['vectors'])
        distances = []
        for v in r['vectors']:
            distances.append(euclidean_distance(centroide, v))
        r['distances'] = distances 
        r.sort_values(by = 'distances')
        questions.append(r['question'].unique().tolist())
    df_cq['size'] = size
    df_cq['questions'] = questions
    df_cq.sort_values(by='size', inplace=True, ascending=False)
    clust_ids = df_cq['clusterId'].unique().tolist()
    NClust = len(clust_ids)
    rec_questions = []
    i = 0
    while len(rec_questions) < num_questions_rec:
        qlist = list(list(df_cq[df_cq['clusterId'] == clust_ids[i%NClust]]['questions'])[0])
        ind = int(np.trunc(i/NClust))
        if ind < len(qlist):
            rec_questions.append(qlist[ind])
        i+=1
    return rec_questions

def list_format(df):
    return df['question'].tolist()