import numpy as np
import os
import skimage
import sys

def load_icoseg_subset_80():
    path = '../datasets/icoseg/subset_80'
    
    train_list = read_img_list(f'{path}/train.txt')
    train = np.array(load_dataset(path, train_list, add_img_format=True))
    
    val_list = read_img_list(f'{path}/val.txt')
    val = np.array(load_dataset(path, val_list, add_img_format=True))
    
    test_list = read_img_list(f'{path}/test.txt')
    test = np.array(load_dataset(path, test_list, add_img_format=True))
    
    return train, val, test

def load_icoseg_subset_80_with_img_names():
    path = '../datasets/icoseg/subset_80'
    
    train_img_names = read_img_list(f'{path}/train.txt')
    train = np.array(load_dataset(path, train_img_names, add_img_format=True))
    
    val_img_names = read_img_list(f'{path}/val.txt')
    val = np.array(load_dataset(path, val_img_names, add_img_format=True))
    
    test_img_names = read_img_list(f'{path}/test.txt')
    test = np.array(load_dataset(path, test_img_names, add_img_format=True))
    
    return train, train_img_names, val, val_img_names, test, test_img_names

def load_dataset(path, img_names=None, add_img_format=False):
    if img_names is None:
        img_names = os.listdir(path + '/images')
    
    images = []
    masks = []
    
    for img_name in img_names:
        img_path = path + '/images/' + img_name
        mask_path = path + '/ground_truth/' + img_name
        
        if add_img_format:
            img_path += '.jpg'
            mask_path += '.png'
            
        img = skimage.io.imread(img_path)
        mask = skimage.io.imread(mask_path)
        
        images.append(img)
        masks.append(mask)
    
    return images, masks

def load_dataset_split(dataset_path, split_name):
    img_path = f'{dataset_path}/{split_name}/{split_name}_img_patches'
    mask_path = f'{dataset_path}/{split_name}/{split_name}_mask_patches'
    
    img_names = os.listdir(img_path)
    images = []
    masks = []
    
    for img_name in img_names:
        img = skimage.io.imread(f'{img_path}/{img_name}')
        mask = skimage.io.imread(f'{mask_path}/{img_name}')
        images.append(img)
        masks.append(mask)
        
    images = np.array(images) / 255.
    masks = np.array(masks) / 255.
    
    return images, masks, img_names

def split_dataset(img_list, train_size=50, val_size=10, test_size=20, random_seed=None):
    np.random.seed(random_seed)
    indices = np.random.choice(len(img_list),
                               train_size + val_size + test_size,
                               replace=False)
    np.random.seed(None)
    
    get_img_names = lambda img_list, start_idx, end_idx:\
                    [img_list[idx] for idx in indices[start_idx:end_idx]]
    
    train = get_img_names(img_list, 0, train_size)
    val = get_img_names(img_list, train_size, train_size + val_size)
    test = get_img_names(img_list, train_size + val_size, train_size + val_size + test_size)
    
    return train, val, test

def write_img_list(img_list, filename):
    with open(filename, 'w') as fo:
        for img_name in img_list:
            fo.write(img_name + '\n')
            
def read_img_list(filename):
    with open(filename, 'r') as fi:
        img_list = fi.read().split()
    return img_list

def create_img_list_from_dir(path):
    full_img_names = os.listdir(path + '/images')
    img_names = []
    
    with open(path + '/img_list.txt', 'w') as fo:
        for full_img_name in full_img_names:
            img_name = full_img_name.split('.')[0]
            fo.write(img_name + '\n')
            img_names.append(img_name)
    
    return img_names

def load_images_from_list(img_names, path):
    images = []
    for img_name in img_names:
        img = skimage.io.imread(f'{path}/{img_name}')
        images.append(img)
        
    images = np.array(images)
    
    return images