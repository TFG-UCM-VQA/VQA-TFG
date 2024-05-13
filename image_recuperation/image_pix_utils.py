from PIL import Image
import numpy as np

from skimage import metrics
#########################################################################
## Similarity mesures                                                  ##
#########################################################################


def pixel_sim(img1, img2, sim_metric):
    comp = 0
    if sim_metric == 'nrmse':
        comp = metrics.normalized_root_mse(img1,img2)
        comp = 1 - comp
    elif sim_metric == 'mse':
        comp = metrics.mean_squared_error(img1,img2)
    elif sim_metric == 'hd':
        comp = metrics.hausdorff_distance(img1,img2)
    elif sim_metric == 'nmi':
        comp = metrics.normalized_mutual_information(img1,img2)
        comp = comp - 1
    elif sim_metric == 'structural':
        #comp = metrics.structural_similarity(img1,img2)
        a = 0
    return comp