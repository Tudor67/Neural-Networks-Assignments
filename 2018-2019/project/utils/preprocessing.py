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
                
def crop_images_from_dir_and_save_all(images_path, save_path,
                                      patch_h, patch_w, img_format):
    img_names = os.listdir(images_path)
    for img_name in img_names:
        img = skimage.io.imread(f'{images_path}/{img_name}')
        img_name_with_shape = append_img_name_with_h_w(remove_img_formats([img_name]),
                                                       get_img_shapes([img]))[0]
        crop_images_and_save([img], [img_name_with_shape],
                             save_path=save_path,
                             img_format=img_format,
                             patch_h=patch_h,
                             patch_w=patch_w)
        
def load_patches(img_name, patches_path):
    patches_names_all = os.listdir(patches_path)
    patches = []
    max_row = 0
    
    for patch_name in sorted(patches_names_all):
        if patch_name.startswith(img_name):
            patch = skimage.io.imread(f'{patches_path}/{patch_name}')
            patches.append(patch)
            
            # useful for patches.reshape
            patch_shape = patch.shape
            row = 1 + int(patch_name.split('_')[-2])
            max_row = max(max_row, row)
            
    patches = np.array(patches).astype(np.uint8).reshape(max_row, -1, *patch_shape)
    return patches                

def get_img_shapes(images):
    return [img.shape for img in images]

def remove_img_formats(img_names):
    return ['.'.join(img_name.split('.')[:-1]) for img_name in img_names]

def remove_grid_indices(img_names):
    return ['_'.join(img_name.split('_')[:-2]) for img_name in img_names]

def append_img_name_with_h_w(img_names, img_shapes):
    return [f'{img_name}_{img_shape[0]}_{img_shape[1]}'
            for img_name, img_shape in zip(img_names, img_shapes)]

def get_img_shapes_from_strings(img_names):
    img_shapes = []
    for img_name in img_names:
        h = int(img_name.split('_')[-4])
        w = int(img_name.split('_')[-3])
        img_shapes.append((h, w))
    return img_shapes

def merge_patches_and_save(img_shapes, img_names, patches_path,
                           save_path, img_format):
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for img_shape, img_name in zip(img_shapes, img_names):
        img_h, img_w = img_shape[:2]
        patches = load_patches(img_name, patches_path)
        img_from_patches = merge_patches(patches, img_h=img_h, img_w=img_w)
        
        filename = f'{save_path}/{img_name}.{img_format}'
        skimage.io.imsave(filename, img_from_patches)

def crop_images_and_save_all(dataset_with_img_names, dataset_path,
                             img_format='png', patch_h=256, patch_w=256, 
                             append_img_h_w=False):
    
    # dataset splits
    train, train_img_names, val, val_img_names, test, test_img_names = dataset_with_img_names
    train_images, train_masks = train
    val_images, val_masks = val
    test_images, test_masks = test
    
    if append_img_h_w:
        train_img_names = append_img_name_with_h_w(train_img_names, get_img_shapes(train_images))
        val_img_names = append_img_name_with_h_w(val_img_names, get_img_shapes(val_images))
        test_img_names = append_img_name_with_h_w(test_img_names, get_img_shapes(test_images))
    
    d_splits = [(train_images, train_img_names, 'train_img'),
                (train_masks, train_img_names, 'train_mask'),
                (val_images, val_img_names, 'val_img'),
                (val_masks, val_img_names, 'val_mask'),
                (test_images, test_img_names, 'test_img'),
                (test_masks, test_img_names, 'test_mask')]
    
    for images, img_names, split_name in d_splits:
        save_path=f'{dataset_path}/{split_name.split("_")[0]}/{split_name}_patches'
    
        crop_images_and_save(images, img_names,
                             save_path=save_path,
                             img_format=img_format,
                             patch_h=patch_h,
                             patch_w=patch_w)
        
def merge_patches_and_save_all(dataset_with_img_names,
                               dataset_path,
                               img_format='png'):
    
    # dataset splits
    train, train_img_names, val, val_img_names, test, test_img_names = dataset_with_img_names
    train_images, train_masks = train
    val_images, val_masks = val
    test_images, test_masks = test
    
    d_splits = [(train_images, train_img_names, 'train_img'),
                (train_masks, train_img_names, 'train_mask'),
                (val_images, val_img_names, 'val_img'),
                (val_masks, val_img_names, 'val_mask'),
                (test_images, test_img_names, 'test_img'),
                (test_masks, test_img_names, 'test_mask')]
    
    for images, img_names, split_name in d_splits:
        patches_path = f'{dataset_path}/{split_name.split("_")[0]}/{split_name}_patches'
        save_path = f'{dataset_path}/{split_name.split("_")[0]}/{split_name}_from_patches'
        
        img_shapes = get_img_shapes(images)
        
        merge_patches_and_save(img_shapes, img_names,
                               patches_path=patches_path,
                               save_path=save_path,
                               img_format=img_format)
        
def merge_patches_directly_and_save_all(results_path,
                                        split_types=['pred'],
                                        img_format='png'):
    
    for split_name in ['train', 'val', 'test']:
        for split_type in split_types:
            patches_path = f'{results_path}/{split_name}/{split_name}_{split_type}_patches'
            save_path = f'{results_path}/{split_name}/{split_name}_{split_type}_from_patches'

            img_names_full = os.listdir(patches_path)
            img_names = remove_grid_indices(img_names_full)        
            img_shapes = get_img_shapes_from_strings(img_names_full)

            merge_patches_and_save(img_shapes, img_names,
                                   patches_path=patches_path,
                                   save_path=save_path,
                                   img_format=img_format)
            
def merge_patches_from_dir_and_save_all(patches_path,
                                        save_path,
                                        img_format='png'):
    
    img_names_full = os.listdir(patches_path)
    img_names = remove_grid_indices(img_names_full)        
    img_shapes = get_img_shapes_from_strings(img_names_full)

    merge_patches_and_save(img_shapes, img_names,
                           patches_path=patches_path,
                           save_path=save_path,
                           img_format=img_format)