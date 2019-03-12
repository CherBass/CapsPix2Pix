# [Image Synthesis with a Convolutional Capsule Generative Adversarial Network (CapsPix2Pix)](https://openreview.net/forum?id=rJen0zC1lE)
## Description
The paper is about using convolutional capsule layers in a conditional GAN framework to synthesise images from binary segmentation labels. The code available here is an implementation of the convolutional capsule GAN used in the paper, and also contains the u-net used for testing the quality of the synthesis. See paper for details.

## Installation
This code requires Python 3, and a Cuda enabled PC to run GPU experiments. It is recomended to install a virtual environment for all the required libraries (listed below).

### Package Requirements 
* pytorch
* torchvision
* numpy
* scipy
* cv2
* matplotlib
* json
* Pillow

### Datasets
The original dataset which was used to train the GAN, can be found in:
https://doi.org/10.5281/zenodo.2559237

An additional prepared dataset has been complied, and can be found in:

This data was used to train u-net, and to train capspix2pix. Download, and place in the same folder as the code to run experiments. Place all .npy in a folder called "npy_data".

## List of prepared data

Training capspix2pix:
* crops256.zip - folder containing 256x256 crops from the original dataset for training capspix2pix. Images are in the "train/original" folder, and labels are in the "train/mask" folder.
* syn256_x_data_val.npy	+ syn256_y_data_val.npy	+ syn256_y_points_data_val.npy (images + labels + centrelines) - validation synthetic dataset, used while training capspix2pix for plotting

Training u-net:
* capspix2pix_AR_data_train.npy	+ capspix2pix_AR_mask_train.npy	(images + labels) - data generated from a capspix2pix model from real labels
* capspix2pix_SSM_data_train.npy + capspix2pix_AR_mask_train.npy	(images + labels) - data generated from a capspix2pix model from synthetic labels
* PBAM_SSM_data_train.npy	+ PBAM_SSM_mask_train.npy	(images + labels) - data generated from PBAM (Physics-based model) for training u-net
* pix2pix_AR_data_train.npy + pix2pix_AR_mask_train.npy	(images + labels) - data generated from a pix2pix model from real labels for training u-net
* pix2pix_SSM_data_train.npy + pix2pix_SSM_mask_train.npy (images + labels) - data generated from a pix2pix model from synthetic labels for training u-net
* real_data_data_train.npy + real_data_mask_train.npy	(images + labels) - augmented real dataset for training u-net

Testing u-net:
* org64_data_test.npy	+ org64_mask_test.npy	(images + labels) - crops from original test dataset for testing u-net

Interpolation:
* crops256_inter_data_train.npy	+ crops256_inter_mask_train.npy	(images + labels) - example data for interpolation


## Code Usage
To start training capspix2pix, first download the datasets as described above and place in the same directory. 

The main code used to train capspix2pix is in train_capspix2pix.py.

The main code used to train u-net is in train_u_net.py.

The capspix2pix generator network is in Capsule_Networks.py, see: capspix2pixG class. The discriminator used in the paper is in Networks.py, see: conditionalCapsDcganD class. See other discriminator options in Capsule_Networks.py.

The class AxonDataset.py is used to load and read from datasets.

For visualisation of the generations and the latent space interpolation, see interpolation.py.

The code for convolutional capsules layers is available in Capsules.py, see: convolutionalCapsule and deconvolutionalCapsule classes.


## License
M.I.T License

## References- if using this code please cite the following paper 
C. Bass, T. Dai, B. Billot, K. Arulkumaran, A. Creswell, C. Clopath, V. De Paola, and A. A.Bharath, “Image synthesis with a convolutional capsule generative adversarial network,” Medical Imaging with Deep Learning, 2019.


