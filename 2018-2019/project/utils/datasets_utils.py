import numpy as np
import os
import skimage
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

def resize_images(images, new_h, new_w, ch):
    resized_images = np.zeros([len(images), new_h, new_w, ch])
    for idx, img in enumerate(images):
        resized_images[idx] = skimage.transform.resize(img,
                                                       [new_h, new_w, ch],
                                                       mode='constant',
                                                       anti_aliasing=False)
    return resized_images