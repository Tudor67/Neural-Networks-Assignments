from keras.layers import *
from keras.models import *
from keras.optimizers import *

from utils import evaluation

def unet(input_size=(256, 256, 3), pretrained_weights=None):
    # common params
    params = {
        'padding': 'same',
        'activation': 'relu',
        'kernel_initializer': 'he_normal'
    }
    
    # input
    inputs = Input(input_size)
    
    # down1
    conv1 = Conv2D(64, 3, **params)(inputs)
    conv1 = Conv2D(64, 3, **params)(conv1)
    pool1 = MaxPooling2D(pool_size=2, strides=2)(conv1)
    
    # down2
    conv2 = Conv2D(128, 3, **params)(pool1)
    conv2 = Conv2D(128, 3, **params)(conv2)
    pool2 = MaxPooling2D(pool_size=2, strides=2)(conv2)
    
    # down3
    conv3 = Conv2D(256, 3, **params)(pool2)
    conv3 = Conv2D(256, 3, **params)(conv3)
    pool3 = MaxPooling2D(pool_size=2, strides=2)(conv3)
    
    # down4
    conv4 = Conv2D(512, 3, **params)(pool3)
    conv4 = Conv2D(512, 3, **params)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=2, strides=2)(drop4)
    
    # bottleneck
    conv5 = Conv2D(1024, 3, **params)(pool4)
    conv5 = Conv2D(1024, 3, **params)(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # up6
    up6 = Conv2D(512, 2, **params)(UpSampling2D(size=2)(drop5))
    concat6 = concatenate([drop4, up6])
    conv6 = Conv2D(512, 3, **params)(concat6)
    conv6 = Conv2D(512, 3, **params)(conv6)
    
    # up7
    up7 = Conv2D(256, 2, **params)(UpSampling2D(size=2)(conv6))
    concat7 = concatenate([conv3, up7])
    conv7 = Conv2D(256, 3, **params)(concat7)
    conv7 = Conv2D(256, 3, **params)(conv7)
    
    # up8
    up8 = Conv2D(128, 2, **params)(UpSampling2D(size=2)(conv7))
    concat8 = concatenate([conv2, up8])
    conv8 = Conv2D(128, 3, **params)(concat8)
    conv8 = Conv2D(128, 3, **params)(conv8)
    
    # up9
    up9 = Conv2D(64, 2, **params)(UpSampling2D(size=2)(conv8))
    concat9 = concatenate([conv1, up9])
    conv9 = Conv2D(64, 3, **params)(concat9)
    conv9 = Conv2D(64, 3, **params)(conv9)
    
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    
    # build the model
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc',
                           evaluation.tf_jaccard,
                           evaluation.tf_dice])
    
    if pretrained_weights is not None:
        model.load_weights(pretrained_weights)
    
    return model