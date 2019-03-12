import os as os
import numpy as np
import matplotlib.pyplot as plt
from tempfile import mkdtemp
import os.path as path

# Shape and save data in numpy form
def shape_data(folder, targets=2):
    images_original = [img for img in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "original"))]
    images_mask = [img for img in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),folder, "mask"))]
    images_box = [img for img in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),folder, "box"))]

    assert (len(images_mask) == len(images_original))
    images_mask.sort()
    images_original.sort()
    images_box.sort()
    print(images_mask)
    print(images_original)


    image = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "original", images_original[0]))

    filename = path.join(mkdtemp(), 'data.dat')
    data = np.memmap(filename, dtype='float64', mode='w+', shape=(len(images_mask), image.shape[0], image.shape[1]))
    filename = path.join(mkdtemp(), 'target.dat')
    target = np.memmap(filename, dtype='float64', mode='w+', shape=(len(images_mask), image.shape[0], image.shape[1]))
    filename = path.join(mkdtemp(), 'target2.dat')
    target2 = np.memmap(filename, dtype='float64', mode='w+', shape=(len(images_mask), image.shape[0], image.shape[1]))

    ctr = 0

    if targets == 3:
        for original_im, labelled_im, labelled2_im in zip(images_original, images_mask, images_box) :
            temp_data = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "original", original_im))
            temp_label = (plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "mask", labelled_im))).astype(float)
            temp_label_2 = (plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "box", labelled2_im))).astype(float)
            data[ctr] = temp_data
            # target has values 0 and 255. make that 0 and 1
            target[ctr] = temp_label
            target2[ctr] = temp_label_2
            ctr += 1

    elif targets == 2:
        for original_im, labelled_im in zip(images_original, images_mask):
            temp_data = plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "original", original_im))
            temp_label = (plt.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), folder, "mask", labelled_im))).astype(float)

            data[ctr] = temp_data
            # target has values 0 and 255. make that 0 and 1
            target[ctr] = temp_label
            ctr += 1

    target = target.astype(int)
    target2 = target2.astype(int)
    print('Shape of data:', data.shape)
    print('Shape of target:', target.shape)
    print('Shape of target2:', target2.shape)

    return data, target, target2


def load_dataset(type, folder='axon_data', name='org', targets=2):
    # Type =  'train' or 'test'
    # folder = directory to read from e.g. 'axon_data'
    # name = prefix to the file name. change to = 'aug' if augmented data
    # save = if you want to save as npy data structure save=1
    # aug = 0 or 1. If 1 then get augemented data + change name

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    mmap_mode = 'r'
    nameData = os.path.join(__location__, 'npy_data/' + name + '_data_' + type + '.npy')
    nameMask = os.path.join(__location__,'npy_data/' + name + '_mask_' + type + '.npy')
    nameBox = os.path.join(__location__,'npy_data/' + name + '_box_' + type + '.npy')
    try:
        X = np.load(nameData, mmap_mode=mmap_mode)
        y = np.load(nameMask, mmap_mode=mmap_mode)
        if targets == 3:
            y2 = np.load(nameBox, mmap_mode=mmap_mode)
        else:
            y2 = []
    except:
        data, mask, box = shape_data(folder +'/'+ type, targets)

        np.save(nameData, data)
        np.save(nameMask, mask)
        np.save(nameBox, box)

        X = np.load(nameData, mmap_mode=mmap_mode)
        y = np.load(nameMask, mmap_mode=mmap_mode)
        if targets == 3:
            y2 = np.load(nameBox, mmap_mode=mmap_mode)
        else:
            y2 = []
        print('data loaded')
    return X, y, y2
