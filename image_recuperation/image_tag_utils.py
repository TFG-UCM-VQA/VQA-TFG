##########################################################################
## Import modules                                                       ##
##########################################################################

import torchvision
import torch
from pycocotools.coco import COCO
import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

##########################################################################
## GENERAL                                                              ##
##########################################################################

def get_ann_tags(imageId, imgsAnns):
    bboxes_real=[]
    catego_real =[]
    for ann in imgsAnns:
        if ann['image_id']==imageId:
            bboxes_real.append(ann['bbox'])
            catego_real.append(ann['category_id'])
    classes_real = [coco_names_RNet[i] for i in catego_real]
    bboxes_real = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in bboxes_real]
    return bboxes_real, classes_real

def get_cat_supercat(coco, cats):
    category_ids = coco.getCatIds()
    categories = coco.loadCats(category_ids)
    supercats = []
    for cat in cats:
        for c in categories:
            if cat == c['name']:
                supercats.append(c['supercategory'])
                break
    return supercats

def box_sizes(boxes):
    sizes = []
    for box in boxes:
        sizes.append((box[0]-box[2])*(box[1]-box[3]))
    Total = sum(sizes)
    sizes = [s/Total for s in sizes]
    return sizes

def draw_boxes(I, boxes, classes, color='r', alpha=0.5):
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Display the image
    
    ax.imshow(I)
    polygons = []

    for i in range(len(boxes)):
            poly = [[boxes[i][0], boxes[i][1]], [boxes[i][0], boxes[i][3]], [boxes[i][2], boxes[i][3]], [boxes[i][2], boxes[i][1]]]
            np_poly = np.array(poly).reshape((4,2))
            polygons.append(Polygon(np_poly))
            ax.text(boxes[i][0], boxes[i][1]-7,classes[i], fontsize=10, color=color, verticalalignment='center')
            
            
    p = PatchCollection(polygons, facecolor=None, edgecolors=color, linewidths=3, alpha=alpha, facecolors=color)
    ax.add_collection(p)
    # Show the image with the rectangle
    #plt.show()
    plt.savefig('img.png', dpi=200, bbox_inches='tight')
    
##########################################################################
## YOLO                                                                 ##
##########################################################################

coco_names_YOLO = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def init_YOLO_model():
    net = cv2.dnn.readNetFromDarknet('../image_recuperation/models/yolov3.cfg', '../image_recuperation/models/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net

def predict_YOLO(image, net, precision):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    net.setInput(blob)
    outputs = net.forward(ln)
    
    boxes = []
    boxes_finales = []

    confidences = []
    confidences_finales = []

    classIDs = []
    classIDs_finales = []
    h, w = image.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if confidences[i] > precision:
                boxes_finales.append([x, y, w+x, h+y])
                classIDs_finales.append(coco_names_YOLO[classIDs[i]])
    return boxes_finales, classIDs_finales


#########################################################################
## RetinaNet                                                           ##
#########################################################################

coco_names_RNet = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def init_RNet_model():
    # download or load the model from disk
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True) #Quizas añadir un min_size
    device = torch.device('cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load the model onto the computation device
    model.eval().to(device)
    return model, device
   
def predict_RNet(image, model, device, detection_threshold):
    # transform the image to tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    with torch.no_grad():
        outputs = model(image) # get the predictions on the image
        
    labels = list(outputs[0]['labels'].detach().cpu().numpy())
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
        
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > detection_threshold]
    # get boxes above the threshold score
    boxes = np.array(bboxes)[np.array(scores) >= detection_threshold].astype(np.int32)
    # get all the predicited class names
    pred_classes = [coco_names_RNet[labels[i]] for i in thresholded_preds_inidices]
    return boxes, pred_classes

#########################################################################
## Similarity mesures                                                  ##
#########################################################################

def set_similarity(tagsA, tagsB):
    if len(tagsA)*len(tagsB) == 0:
        return 0
    tagsA = list(dict.fromkeys(tagsA))
    tagsB = list(dict.fromkeys(tagsB))
    AintB = 0
    for tag in tagsA:
        if tag in tagsB:
            AintB += 1
    return AintB/(len(tagsA)+len(tagsB)-AintB)

def multiset_similarity(tagsA, tagsB):
    if len(tagsA)*len(tagsB) == 0:
        return 0
    all_tags = list(dict.fromkeys(tagsA+tagsB))
    AintB = 0
    for tag in all_tags:
        nA = sum([1 if t == tag else 0 for t in tagsA])
        nB = sum([1 if t == tag else 0 for t in tagsB])
        AintB += min(nA, nB)
    return AintB/(len(tagsA)+len(tagsB)-AintB)
    
def set_similarity_s(tagsA, tagsB, coco, h):
    if len(tagsA)*len(tagsB) == 0:
        return 0
    tagsA = list(dict.fromkeys(tagsA))
    tagsB = list(dict.fromkeys(tagsB))
    supA = get_cat_supercat(coco, tagsA)
    supB = get_cat_supercat(coco, tagsB)
    AintB = 0
    AintBs = 0
    for i in range(len(tagsA)):
        if tagsA[i] in tagsB:
            AintB += 1
        elif supA[i] in supB:
            AintBs += 1
    AintB += AintBs*h
    return AintB/(len(tagsA)+len(tagsB)-AintB)

def multiset_similarity_s(tagsA, tagsB, coco, h):
    if len(tagsA)*len(tagsB) == 0:
        return 0
    supA = get_cat_supercat(coco, tagsA)
    supB = get_cat_supercat(coco, tagsB)
    all_tags = list(dict.fromkeys(tagsA+tagsB))
    all_supt = list(dict.fromkeys(supA+supB))
    AintB = 0
    AintBs = 0
    for tag in all_tags:
        nA = sum([1 if t == tag else 0 for t in tagsA])
        nB = sum([1 if t == tag else 0 for t in tagsB])
        AintB += min(nA, nB)
    for tag in all_supt:
        nA = sum([1 if t == tag else 0 for t in supA])
        nB = sum([1 if t == tag else 0 for t in supB])
        AintBs += min(nA, nB)
    AintBs -= AintB
    AintB += AintBs*h
    return AintB/(len(tagsA)+len(tagsB)-AintB)

def size_set_similarity(tagsA, tagsB, sizeA, sizeB):
    all_tags = list(dict.fromkeys(tagsA+tagsB))
    AintB = 0
    for tag in all_tags:
        nA = sum([sizeA[i] if tagsA[i] == tag else 0 for i in range(len(tagsA))])
        nB = sum([sizeB[i] if tagsB[i] == tag else 0 for i in range(len(tagsB))])
        AintB += min(nA,nB)
    return AintB

def size_set_similarity_s(tagsA, tagsB, sizeA, sizeB, coco, h):
    supA = get_cat_supercat(coco, tagsA)
    supB = get_cat_supercat(coco, tagsB)
    all_tags = list(dict.fromkeys(tagsA+tagsB))
    all_supt = list(dict.fromkeys(supA+supB))
    AintB = 0
    AintBs = 0
    for tag in all_tags:
        nA = sum([sizeA[i] if tagsA[i] == tag else 0 for i in range(len(tagsA))])
        nB = sum([sizeB[i] if tagsB[i] == tag else 0 for i in range(len(tagsB))])
        AintB += min(nA,nB)
    for tag in all_supt:
        nA = sum([sizeA[i] if supA[i] == tag else 0 for i in range(len(supA))])
        nB = sum([sizeB[i] if supB[i] == tag else 0 for i in range(len(supB))])
        AintBs += min(nA,nB)
    return AintB + h*(AintBs - AintB)