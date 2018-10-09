# Get MNIST
rm -rf ../datasets/mnist
mkdir ../datasets/mnist

wget http://yann.lecun.com/exdb/mnist/index.html
mv index.html ../datasets/mnist/readme.html

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
mv train-images-idx3-ubyte ../datasets/mnist/train_images

wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz 
mv train-labels-idx1-ubyte ../datasets/mnist/train_labels

wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
mv  t10k-images-idx3-ubyte ../datasets/mnist/test_images

wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
mv t10k-labels-idx1-ubyte ../datasets/mnist/test_labels
