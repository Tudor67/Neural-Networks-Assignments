import numpy as np
import skimage

from collections import deque
from params import *

def preprocess_frame(rgb_frame):
    # gray
    gray_frame = skimage.color.rgb2gray(rgb_frame)
    
    # crop
    cropped_frame = gray_frame[8:-12, 4:-12]
    
    # norm
    norm_frame = cropped_frame / 255.
    
    # resize
    preprocessed_frame = skimage.transform.resize(norm_frame, [110, 84], 
                                                  mode='constant',
                                                  anti_aliasing=False)
    
    # preprocessed
    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames = deque(maxlen=STACK_SIZE)
        for _ in range(STACK_SIZE):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    
    stacked_states = np.moveaxis(stacked_frames, 0, -1)
    
    # stacked_states.shape: [110, 84, 4]
    # stacked_frames.shape: [4, 110, 84]
    return stacked_states, stacked_frames