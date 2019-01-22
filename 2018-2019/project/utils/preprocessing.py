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

def crop_image(img, patch_h=256, patch_w=256):
    patch_shape = (patch_h, patch_w, 3)
    if img.ndim == 2:
        img = img[:,:,np.newaxis]
        patch_shape = (patch_h, patch_w, 1)
        
    row_pad = (patch_shape[0] - (img.shape[0] % patch_shape[0])) % patch_shape[0]
    col_pad = (patch_shape[1] - (img.shape[1] % patch_shape[1])) % patch_shape[1]
    
    img_pad = np.pad(img, [(0, row_pad), (0, col_pad), (0, 0)], 'constant')
    
    rows_start = range(0, img_pad.shape[0], patch_shape[0])
    cols_start = range(0, img_pad.shape[1], patch_shape[1])
    
    patches = np.zeros([len(rows_start), len(cols_start), *patch_shape],
                       dtype=np.uint8)
    
    for i, row in enumerate(rows_start):
        for j, col in enumerate(cols_start):
            patches[i][j] = img_pad[row:row + patch_shape[0],
                                    col:col + patch_shape[1],
                                    :]
    
    return patches.squeeze()

def merge_patches(patches, img_h, img_w):
    img_pad_h = patches.shape[0] * patches.shape[2]
    img_pad_w = patches.shape[1] * patches.shape[3]
    
    # combine patches
    patches = np.moveaxis(patches, 2, 1)
    img_pad = patches.reshape([img_pad_h, img_pad_w, -1])
    
    # remove padding
    img = img_pad[:img_h, :img_w, :].squeeze()
    
    return img