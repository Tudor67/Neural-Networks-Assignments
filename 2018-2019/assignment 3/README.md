# Assignment 3 (2018-2019)

## TensorFlow results
1. `ConvNet_CIFAR10_TensorFlow.ipynb`
2. `$ tensorboard --logdir=./logs`

| Dataset split | Initialization method | Loss | Accuracy | Epoch |
| :--- | :---: | :---: | :---: | :---: |
| Train (random 1,000) | he_normal | 0.72 | 75.50% | 15 |
| Validation (1,000) | he_normal | 1.32 | 62.50% | 15 |
| Test (random 1,000) | he_normal | 1.36 | 60.80% | 15 |
| Train (random 1,000) | he_uniform | 0.65 | 78.10% | 15 |
| Validation (1,000) | he_uniform | 1.22 | 63.30% | 15 |
| Test (random 1,000) | he_uniform | 1.26 | 60.30% | 15 |
| Train (random 1,000) | glorot_normal | 0.61 | 78.70% | 15 |
| Validation (1,000) | glorot_normal | 1.45 | 63.70% | 15 |
| Test (random 1,000) | glorot_normal | 1.58 | 61.50% | 15 |
| Train (random 1,000) | glorot_uniform | 0.66 | 77.10% | 15 |
| Validation (1,000) | glorot_uniform | 1.23 | 64.10% | 15 |
| Test (random 1,000) | glorot_uniform | 1.37 | 62.50% | 15 |

### Conv1 weights/filters
The initialization method of the weights: `he_normal` and `he_uniform`. These weights are ugly. I expected to see edge or color filters, not random values.
![conv1_weights_init_he_normal_tensorflow](./images/conv1_weights_init_he_normal_tensorflow.png)
![conv1_weights_init_he_uniform_tensorflow](./images/conv1_weights_init_he_uniform_tensorflow.png)

The initialization method of the weights: `glorot_normal` and `glorot_uniform`. These weights look a little better.
![conv1_weights_init_glorot_normal_tensorflow](./images/conv1_weights_init_glorot_normal_tensorflow.png)
![conv1_weights_init_glorot_uniform_tensorflow](./images/conv1_weights_init_glorot_uniform_tensorflow.png)

It is interesting to see the distributions of the weights for each initialization method. We can observe some differences just for the weights from the first convolutional layer.
![weights_distributions_tensorflow](./images/weights_distributions_tensorflow.png)

## PyTorch results
1. `ConvNet_CIFAR10_PyTorch.ipynb`

| Dataset split | Initialization method | Accuracy | Epoch |
| :--- | :---: | :---: | :---: |
| Train (49,000) | glorot_normal | 83.49% | 15 |
| Validation (1,000) | glorot_normal | 69.00% | 15 |
| Test (10,000) | glorot_normal | 67.42% | 15 |

### Conv1 weights/filters
The initialization method of the weights: `glorot_normal`.
![conv1_weights_pytorch](./images/conv1_weights_pytorch.png)
