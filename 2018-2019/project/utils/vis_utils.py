import matplotlib.pyplot as plt
import numpy as np

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