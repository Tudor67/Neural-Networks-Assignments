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