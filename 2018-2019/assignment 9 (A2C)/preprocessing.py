import config
import numpy as np
import skimage

from collections import deque

def preprocess_frame(rgb_frame):
    # gray
    gray_frame = skimage.color.rgb2gray(rgb_frame)
    
    # resize
    resized_frame = skimage.transform.resize(gray_frame,
                                             [config.FRAME_H, config.FRAME_W], 
                                             mode='constant',
                                             anti_aliasing=False)
    return resized_frame

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames = deque(maxlen=config.STACK_SIZE+1)
        for _ in range(config.STACK_SIZE):
            stacked_frames.append(frame)
    else:
        stacked_frames.append(frame)
    stacked_frames_hwc = np.moveaxis(stacked_frames, 0, -1)
    
    # stacked_frames_hwc.shape: [84, 84, 4(+1 if is not a new episode)]
    # stacked_frames.shape: [4(+1 if is not a new episode), 84, 84]
    return stacked_frames_hwc, stacked_frames