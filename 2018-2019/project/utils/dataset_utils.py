import numpy as np
import os
import sys

def load_dataset(path):
    img_names = os.listdir(path + '/images')
    
    images = []
    masks = []
    
    for img_name in img_names:
        img = skimage.io.imread(path + '/images/' + img_name)
        mask = skimage.io.imread(path + '/ground_truth/' + img_name)
        
        images.append(img)
        masks.append(mask)
    
    return images, masks

def create_img_list(path):
    full_img_names = os.listdir(path + '/images')
    with open(path + '/img_list.txt', 'w') as fo:
        for full_img_name in full_img_names:
            img_name = full_img_name.split('.')[0]
            fo.write(img_name + '\n')