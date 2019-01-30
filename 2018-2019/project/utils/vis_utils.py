import matplotlib.pyplot as plt
import numpy as np

def grid_vis(images, grid_rows=2, grid_cols=2, plt_axis='off',
             grid_col_dim=5, grid_row_dim=5, wspace=0.075, hspace=0.075):
    plt.figure(figsize=(grid_cols * grid_col_dim, grid_rows * grid_row_dim))
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    
    for i in range(grid_rows):
        for j in range(grid_cols):
            img_idx = i * grid_cols + j
            plt.subplot(grid_rows, grid_cols, img_idx + 1)
            plt.axis(plt_axis)
            plt.imshow(images[img_idx], cmap='gray')

def grid_vis_for_crop_and_merge(img, img_patches, mask, mask_patches):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title('image')
    plt.axis('off')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.title('mask')
    plt.axis('off')
    plt.imshow(mask, cmap='gray')

    img_patches_num = img_patches.shape[0] * img_patches.shape[1]
    plt.figure(figsize=(12.5, 5))

    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            plt.subplot(img_patches.shape[0],
                        2 * img_patches.shape[1],
                        i * img_patches_num + j + 1)
            plt.title(f'img_patches[{i}][{j}]')
            plt.axis('off')
            plt.imshow(img_patches[i][j])

            plt.subplot(img_patches.shape[0],
                        2 * img_patches.shape[1],
                        i * img_patches_num + img_patches.shape[1] + j + 1)

            plt.title(f'mask_patches[{i}][{j}]')
            plt.axis('off')
            plt.imshow(mask_patches[i][j], cmap='gray')
            
def plot_image_mask_prediction(images, masks, preds, dataset_split='', pred_thr=None):
    print(f'dataset_split: {dataset_split}')
    
    num_cols = 4
    fig_size = (10, 2)
    if pred_thr is not None:
        num_cols = 5
        fig_size = (12, 2)
        
    for idx in range(len(preds)):
        plt.figure(figsize=fig_size)
        plt.subplot(1, num_cols, 1)
        plt.title(f'initial image ({dataset_split})')
        plt.axis('off')
        plt.imshow(images[idx])
        plt.subplot(1, num_cols, 2)
        plt.title('ground truth')
        plt.axis('off')
        plt.imshow(masks[idx].squeeze(), cmap='gray')
        plt.subplot(1, num_cols, 3)
        plt.title('prediction')
        plt.axis('off')
        plt.imshow(preds[idx].squeeze(), cmap='gray')
        plt.subplot(1, num_cols, 4)
        plt.title('prediction values')
        plt.axis('on')
        plt.hist(preds[idx].flatten(), density=True)
        
        if pred_thr is not None:
            plt.subplot(1, num_cols, 5)
            plt.title(f'prediction (thr={pred_thr})')
            plt.axis('off')
            plt.imshow(preds[idx].squeeze() > pred_thr, cmap='gray')
        
        plt.show()
        
def plot_train_val(train, val, fig_title='Loss', y_label='binary cross entropy'):
    x = 1 + np.arange(len(train))
    plt.figure(figsize=(8, 5))
    plt.title(f'{fig_title} evolution')
    plt.plot(x, train, label=f'train {fig_title.lower()}')
    plt.plot(x, val, label=f'val {fig_title.lower()}')
    plt.ylabel(y_label)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()