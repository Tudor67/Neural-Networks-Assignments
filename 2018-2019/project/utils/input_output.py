import json
import os
import skimage

def read_json(filename):
    with open(filename, 'r') as fi:
        data = json.load(fi)
    return data

def write_json(data, filename):
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        
    with open(filename, 'w') as fo:
        json.dump(data, fo)
        
def tiff_to_png(in_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    
    img_names = os.listdir(in_path)
    
    for img_name in img_names:
        img = skimage.io.imread(f'{in_path}/{img_name}')
        img_name_png = '.'.join(img_name.split('.')[:-1]) + '.png'
        skimage.io.imsave(f'{out_path}/{img_name_png}', img)
        
def rename_images_with_uniform_ind_len(dir_path):
    img_names = sorted(os.listdir(dir_path))

    for img_name in img_names:
        img_name_pref = '_'.join(img_name.split('_')[:-2])
        img_format = img_name.split('.')[-1]

        i = img_name.split('_')[-2]
        if len(i) < 2:
            i = '0' + i

        j = img_name.split('_')[-1].split('.')[0]
        if len(j) < 2:
            j = '0' + j

        new_img_name = f'{img_name_pref}_{i}_{j}.{img_format}'

        os.rename(f'{dir_path}/{img_name}',
                  f'{dir_path}/{new_img_name}')