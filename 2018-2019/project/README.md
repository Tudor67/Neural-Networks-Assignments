# Project (2018-2019)
__Image segmentation with saliency maps__

## Poster
<object data="https://github.com/Tudor67/Neural-Networks-Assignments/blob/master/2018-2019/project/presentations/Tudor_Buzu_NN_Poster.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/Tudor67/Neural-Networks-Assignments/blob/master/2018-2019/project/presentations/Tudor_Buzu_NN_Poster.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/Tudor67/Neural-Networks-Assignments/blob/master/2018-2019/project/presentations/Tudor_Buzu_NN_Poster.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Installation
1. Python 3.6
2. TensorFlow (1.11)
3. Keras (2.2.4)
4. keras-vis (last version)
    - `pip install git+https://github.com/raghakot/keras-vis.git`

## Datasets
### __CMU-Cornell iCoseg dataset__
1. Look at some groups of images: http://chenlab.ece.cornell.edu/projects/touch-coseg/iCoseg_dataset.pdf;
2. Download the dataset from: http://chenlab.ece.cornell.edu/downloads.html;
3. Move dataset folders inside the `./datasets/`;
4. Copy 80 images and their ground truths from `./datasets/icoseg/images/...` and `./datasets/icoseg/ground_truth/...` to `./datasets/icoseg/subset_80/images` and `./datasets/icoseg/subset_80/grund_truth`.
The list of images to be copied: `./datasets/icoseg/subset_80/img_list.txt`.  

Folder structure:

    .
    ├── ...
    ├── datasets
    │   ├── icoseg
    │   │   ├── ground_truth
    │   │   ├── image_download
    │   │   ├── images
    │   │   ├── images_subset5_cvpr10
    │   │   ├── subset_80
    │   │   │   ├── ground_truth
    │   │   │   ├── images
    │   │   │   ├── img_list.txt
    │   │   │   ├── train.txt
    │   │   │   ├── val.txt
    │   │   │   └── test.txt
    │   │   ├── readme.txt
    │   │   └── ...
    │   └── ...
    └── ...


## References
1. [Visualizing and Understanding Convolutional Networks (Zeiler and Fergus, 2013)](https://arxiv.org/abs/1311.2901)
2. [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps (Simonyan _et al._, 2014)](https://arxiv.org/abs/1312.6034)
3. [Striving for Simplicity: The All Convolutional Net (Springenberg _et al._, 2015)](https://arxiv.org/abs/1412.6806)
4. [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (Selvaraju _et al._, 2016)](https://arxiv.org/abs/1610.02391)
5. [iCoseg: Interactive Co-segmentation with Intelligent Scribble Guidance (Dhruv Batra _et al_, 2010)](https://www.researchgate.net/publication/224164344_iCoseg_Interactive_co-segmentation_with_intelligent_scribble_guidance)
6. [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger _et al._, 2015)](https://arxiv.org/abs/1505.04597)
7. [keras-vis for neural network visualization](https://raghakot.github.io/keras-vis/visualizations/saliency/)
8. https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/

