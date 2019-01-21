import numpy as np
import os
import skimage
import sys

def resize_images(images, new_h, new_w, ch):
    resized_images = np.zeros([len(images), new_h, new_w, ch])
    for idx, img in enumerate(images):
        resized_images[idx] = skimage.transform.resize(img,
                                                       [new_h, new_w, ch],
                                                       mode='constant',
                                                       anti_aliasing=False)
    return resized_images