import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def save_ani(episode, reward, frames, out_path='./results/', mode='train'):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    fig = plt.figure()
    plt.title(f'{mode} episode: {episode}, reward: {reward}')
    plt.axis('off')

    im = plt.imshow(frames[0], animated=True)

    def update_fig(frame, *args):
        im.set_array(frame)
        return im,

    ani = animation.FuncAnimation(fig, update_fig, frames=frames[::5])
    ani.save(f'{out_path}{mode}_episode_{episode}_reward_{reward}.gif', 
             writer='imagemagick',
             fps=50)

def save_train_ani(dir_name='./train_stats'):
    filenames = os.listdir(dir_name)
    print(filenames)

    for filename in filenames:
        frames = np.load(f'./train_stats/{filename}')
        episode = int(filename.split('_')[2].split('.')[0])
        reward = int(filename.split('_')[4].split('.')[0])
        save_ani(episode, reward, frames, out_path='./results/', mode='train')