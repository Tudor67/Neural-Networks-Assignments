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

def create_img_list_from_dir(path):
    full_img_names = os.listdir(path + '/images')
    img_names = []
    
    with open(path + '/img_list.txt', 'w') as fo:
        for full_img_name in full_img_names:
            img_name = full_img_name.split('.')[0]
            fo.write(img_name + '\n')
            img_names.append(img_name)
    
    return img_names