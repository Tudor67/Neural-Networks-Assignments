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
    if patches.shape[4] == 1:
        patches = patches.squeeze(axis=4)
        
    return patches

def merge_patches(patches, img_h, img_w):
    img_pad_h = patches.shape[0] * patches.shape[2]
    img_pad_w = patches.shape[1] * patches.shape[3]
    
    # combine patches
    patches = np.moveaxis(patches, 2, 1)
    img_pad = patches.reshape([img_pad_h, img_pad_w, -1])
    
    # remove padding
    img = img_pad[:img_h, :img_w, :].squeeze()
    
    return img

def crop_images_and_save(images, img_names,
                         save_path, img_format,
                         patch_h, patch_w):
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for img, img_name in zip(images, img_names):
        img_patches = crop_image(img, patch_h, patch_w)
        
        for i in range(img_patches.shape[0]):
            for j in range(img_patches.shape[1]):
                filename = f'{save_path}/{img_name}_{i}_{j}.{img_format}'
                skimage.io.imsave(filename, img_patches[i][j])
                
                
def crop_images_and_save_all(dataset_with_img_names, dataset_path,
                             img_format='png', patch_h=256, patch_w=256):
    
    train, train_img_names, val, val_img_names, test, test_img_names = dataset_with_img_names
    train_images, train_masks = train
    val_images, val_masks = val
    test_images, test_masks = test
    
    # train split
    crop_images_and_save(train_images, train_img_names,
                         save_path=f'{dataset_path}/train_img_patches',
                         img_format=img_format, patch_h=patch_h, patch_w=patch_w)
    crop_images_and_save(train_masks, train_img_names,
                         save_path=f'{dataset_path}/train_mask_patches',
                         img_format=img_format, patch_h=patch_h, patch_w=patch_w)
    # val split
    crop_images_and_save(val_images, val_img_names,
                         save_path=f'{dataset_path}/val_img_patches',
                         img_format=img_format, patch_h=patch_h, patch_w=patch_w)
    crop_images_and_save(val_masks, val_img_names,
                         save_path=f'{dataset_path}/val_mask_patches',
                         img_format=img_format, patch_h=patch_h, patch_w=patch_w)
    # test split
    crop_images_and_save(test_images, test_img_names,
                         save_path=f'{dataset_path}/test_img_patches',
                         img_format=img_format, patch_h=patch_h, patch_w=patch_w)
    crop_images_and_save(test_masks, test_img_names,
                         save_path=f'{dataset_path}/test_mask_patches',
                         img_format=img_format, patch_h=patch_h, patch_w=patch_w)