import numpy as np
import random

# pytorch- initialising weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#Cropping images for training
def random_crop(inputs, targets, num_targets=1, width=128, height=128):
    "Create random crop from an image and its targets"
    #inputs = numpy array of shape [row, col]
    #targets = numpy array of targets [row, col] or [row, col][row, col]
    #num_targets = number of targets
    #width = width of crop
    #height = height of crop

    row = inputs.shape[0]
    col = inputs.shape[1]

    blank = 1
    while blank:
        x = random.randint(0, row-width-1)
        y = random.randint(0, col-height-1)

        input_crop = inputs[x:x+width, y:y+height]
        if num_targets == 1:
            target_crop = targets[x:x+width, y:y+height]
            if np.max(target_crop) == 1:
                blank = 0
        elif num_targets == 2:
            target1 = targets[0]
            target2 = targets[1]
            target1 = target1[x:x+width, y:y+height]
            target2 = target2[x:x+width, y:y+height]
            target_crop = [target1, target2]
            if np.max(target1) == 1 or np.max(target2) == 1:
                blank = 0
    return input_crop, target_crop
