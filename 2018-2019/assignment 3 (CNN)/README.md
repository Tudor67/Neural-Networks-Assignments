# Assignment 3 (2018-2019)

## Architecture
1.1) Conv(5, 5, filters=64)  
1.2) ReLU  
1.3) MaxPooling(ksize=2, stride=2)  
2.1) Conv(5, 5, filters=16)  
2.2) ReLU  
2.3) MaxPooling(ksize=2, stride=2)  
3.1) FC(200)  
3.2) ReLU  
4.1) FC(10)  
4.2) Softmax

## Training details
* train data: 49,000 images;
* validation data: 1,000 images;
* epochs: 15;
* batch_size: 64;
* loss: cross-entropy;
* optimizer: Adam with default parameters.

## TensorFlow results
1. `ConvNet_CIFAR10_TensorFlow.ipynb`
2. `$ tensorboard --logdir=./logs`

| Dataset split | Initialization method | Loss | Accuracy | Epoch |
| :--- | :---: | :---: | :---: | :---: |
| Train (random 1,000) | he_normal | 0.72 | 75.50% | 15 |
| Validation (1,000) | he_normal | 1.32 | 62.50% | 15 |
| Train (random 1,000) | he_uniform | 0.65 | 78.10% | 15 |
| Validation (1,000) | he_uniform | 1.22 | 63.30% | 15 |
| Train (random 1,000) | glorot_normal | 0.61 | 78.70% | 15 |
| Validation (1,000) | glorot_normal | 1.45 | 63.70% | 15 |
| Train (random 1,000) | glorot_uniform | 0.66 | 77.10% | 15 |
| Validation (1,000) | glorot_uniform | 1.23 | 64.10% | 15 |

### Conv1 weights/filters
Initialization methods of the weights: `he_normal` and `he_uniform`. These weights are ugly. I expected to see edge or color filters, not random values.
![conv1_weights_init_he_normal_tensorflow](./images/conv1_weights_init_he_normal_tensorflow.png)
![conv1_weights_init_he_uniform_tensorflow](./images/conv1_weights_init_he_uniform_tensorflow.png)

Initialization methods of the weights: `glorot_normal` and `glorot_uniform`. These weights look a little better.
![conv1_weights_init_glorot_normal_tensorflow](./images/conv1_weights_init_glorot_normal_tensorflow.png)
![conv1_weights_init_glorot_uniform_tensorflow](./images/conv1_weights_init_glorot_uniform_tensorflow.png)

It is interesting to see the distributions of the weights for each initialization method. We can observe some differences just for the weights from the first convolutional layer.
![weights_distributions_tensorflow](./images/weights_distributions_tensorflow.png)

## PyTorch results
1. `ConvNet_CIFAR10_PyTorch.ipynb`

| Dataset split | Initialization method | Accuracy | Epoch |
| :--- | :---: | :---: | :---: |
| Train (49,000) | he_normal | 76.27% | 15 |
| Validation (1,000) | he_normal | 66.60% | 15 |
| Train (49,000) | he_uniform | 77.19% | 15 |
| Validation (1,000) | he_uniform | 65.60% | 15 |
| Train (49,000) | glorot_normal | 73.06% | 15 |
| Validation (1,000) | glorot_normal | 67.00% | 15 |
| Train (49,000) | glorot_normal | 82.17% | 15 |
| Validation (1,000) | glorot_normal | 69.10% | 15 |

### Conv1 weights/filters
Initialization methods of the weights: `he_normal` and `he_uniform`. These weights are ugly.
![conv1_weights_init_he_normal_pytorch](./images/conv1_weights_init_he_normal_pytorch.png)
![conv1_weights_init_he_uniform_pytorch](./images/conv1_weights_init_he_uniform_pytorch.png)

Initialization methods of the weights: `glorot_normal` and `glorot_uniform`. These weights look a little better.
![conv1_weights_init_glorot_normal_pytorch](./images/conv1_weights_init_glorot_normal_pytorch.png)
![conv1_weights_init_glorot_uniform_pytorch](./images/conv1_weights_init_glorot_uniform_pytorch.png)
