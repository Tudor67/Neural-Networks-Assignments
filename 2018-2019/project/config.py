BACKPROP_MODIFIER = 'None'
UNET_NAME = f'UNet_RGB_SAL_MAP_{BACKPROP_MODIFIER.upper()}'
#UNET_NAME = f'UNet_RGB'
#DATASET_NAME = 'icoseg/subset_80'
DATASET_NAME = 'inria_aerial/subset_chicago'
EXPERIMENT_NAME = f'{DATASET_NAME}/{UNET_NAME}'

PRED_THR = 0.5
#INPUT = 'sal_map'
#INPUT = 'rgb'
INPUT_SHAPE = (512, 512, 4)
SALIENCY_INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 1
NUM_EPOCHS = 20