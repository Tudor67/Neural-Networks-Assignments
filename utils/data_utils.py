from six.moves import cPickle as pickle

import array
import functools
import numpy as np
import operator
import os
import platform
import struct

MNIST_DOWNLOAD_SCRIPT = '../datasets/download_mnist.sh'
MNIST_DIR = '../datasets/mnist/'
MNIST_TRAIN_IMAGES = MNIST_DIR + 'train_images'
MNIST_TRAIN_LABELS = MNIST_DIR + 'train_labels'
MNIST_TEST_IMAGES = MNIST_DIR + 'test_images'
MNIST_TEST_LABELS = MNIST_DIR + 'test_labels'

CIFAR10_DOWNLOAD_SCRIPT = '../datasets/download_cifar10.sh'
CIFAR10_DIR = '../datasets/cifar10'


def download_mnist():
    os.system(MNIST_DOWNLOAD_SCRIPT)


def download_cifar10():
    os.system(CIFAR10_DOWNLOAD_SCRIPT)

    
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
        
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    
    return Xtr, Ytr, Xte, Yte


def get_cifar10_data(num_training=49000, num_validation=1000, num_test=1000, shuffle=False):
    """
    Load the CIFAR-10 dataset
    """
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load_CIFAR10(CIFAR10_DIR)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask] / 255.0
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask] / 255.0
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask] / 255.0
    y_test = y_test[mask]
    
    if shuffle:
        indices_train = np.arange(num_training)
        np.random.shuffle(indices_train)
        X_train = X_train[indices_train]
        y_train = y_train[indices_train]
        
        indices_val = np.arange(num_validation)
        np.random.shuffle(indices_val)
        X_val = X_val[indices_val]
        y_val = y_val[indices_val]
        
        indices_test = np.arange(num_test)
        np.random.shuffle(indices_test)
        X_test = X_test[indices_test]
        y_test = y_test[indices_test]
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    classes = ['plane', 'car', 'bird', 'cat', 'deer',\
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
      'classes': classes,
    }


def parse_idx(fname):
    """
    From https://github.com/datapythonista/mnist/blob/master/mnist/__init__.py
    Parse an IDX file, and return it as a numpy array.
    Parameters
    ----------
    fname : str
        Filename of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options
    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html
        #byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)
    
    with open(fname, 'rb') as fd:
        header = fd.read(4)
        if len(header) != 4:
            raise IdxDecodeError('Invalid IDX file, '
                                 'file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise IdxDecodeError('Invalid IDX file, '
                                 'file must start with two zero bytes. '
                                 'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise IdxDecodeError('Unknown data type '
                                 '0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise IdxDecodeError('IDX file has wrong number of items. '
                                 'Expected: %d. Found: %d' % (expected_items,
                                                              len(data)))

        return np.array(data).reshape(dimension_sizes)


def load_MNIST():
    X_train = parse_idx(MNIST_TRAIN_IMAGES)
    y_train = parse_idx(MNIST_TRAIN_LABELS)
    X_test = parse_idx(MNIST_TEST_IMAGES)
    y_test = parse_idx(MNIST_TEST_LABELS)
    return (X_train, y_train, X_test, y_test)


def get_mnist_data(num_training=49000, num_validation=1000, num_test=1000, shuffle=False):
    """
    Load the MNIST dataset
    """
    # Load the raw CIFAR-10 data
    mnist_dir = '../datasets/mnist'
    X_train, y_train, X_test, y_test = load_MNIST()
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask] / 255.0
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask] / 255.0
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask] / 255.0
    y_test = y_test[mask]
    
    if shuffle:
        indices_train = np.arange(num_training)
        np.random.shuffle(indices_train)
        X_train = X_train[indices_train]
        y_train = y_train[indices_train]
        
        indices_val = np.arange(num_validation)
        np.random.shuffle(indices_val)
        X_val = X_val[indices_val]
        y_val = y_val[indices_val]
        
        indices_test = np.arange(num_test)
        np.random.shuffle(indices_test)
        X_test = X_test[indices_test]
        y_test = y_test[indices_test]
        
    # Transpose so that channels come first
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()
    
    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }