# [Image Synthesis with a Convolutional Capsule Generative Adversarial Network (CapsPix2Pix)](https://openreview.net/forum?id=rJen0zC1lE)
## Description
The paper is about using convolutional capsule layers in a conditional GAN framework to synthesise images from binary segmentation labels.

## Installation
This code requires Python 3, and a Cuda enabled PC to run GPU experiments.

### Requirements 
<pytorch>
torchvision
numpy
scipy
cv2
matplotlib
json

## Datasets
The original dataset which was used to train the GAN, can be found in:
https://doi.org/10.5281/zenodo.2559237

An additional prepared dataset has been complied, and can be found in:

This data was used to train u-net, and to train capspix2pix. Download, and place in the same folder as the code to run experiments.

## License
M.I.T License

## References- if using this code please cite the following paper 
C. Bass, T. Dai, B. Billot, K. Arulkumaran, A. Creswell, C. Clopath, V. De Paola, and A. A.Bharath, “Image synthesis with a convolutional capsule generative adversarial network,” Medical Imaging with Deep Learning, 2019.


