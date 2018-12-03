import matplotlib.pyplot as plt
import numpy as np

def grid_vis(X, img_h=28, img_w=28, img_c=1, pad=1, norm_img=True,
             rows=5, cols=15, plt_title='', figsize=None):
    
    X = X[:rows*cols]
    
    if norm_img:
        X_min = X.min(axis=tuple(range(1, X.ndim)), keepdims=True)
        X_max = X.max(axis=tuple(range(1, X.ndim)), keepdims=True)
        X = (X - X_min) / (X_max - X_min)
        
    if figsize is None:
        figsize=(16, 5)
    
    X_reshaped = X.reshape(-1, img_h, img_w, img_c)
    
    g_d1 = rows
    g_d2 = img_h + 2 * pad
    g_d3 = g_d2 * cols
    g_d4 = img_c
    grid_rows = np.zeros((g_d1, g_d2, g_d3, g_d4))
        
    for i in range(rows):
        start_idx = i * cols
        end_idx = (i + 1) * cols
        X_pad = np.pad(X_reshaped[start_idx:end_idx], 
                       [(0,), (pad,), (pad,), (0,)], 'constant', constant_values=1)
        grid_rows[i] = np.concatenate(X_pad, axis=1)
    
    grid_imgs = np.concatenate(grid_rows, axis=0)
    plt.figure(figsize=figsize)
    plt.title(plt_title)
    plt.imshow(grid_imgs.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    
def plot_loss_evolution(D_loss, G_loss):
    plt.figure()
    plt.title('Loss')
    plt.plot(range(1, len(D_loss)), D_loss[1:], label='D_loss')
    plt.plot(range(1, len(G_loss)), G_loss[1:], label='G_loss')
    plt.ylabel('x_entropy loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()