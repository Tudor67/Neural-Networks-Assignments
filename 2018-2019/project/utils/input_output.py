import json
import os

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