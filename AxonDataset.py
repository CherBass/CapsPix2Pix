import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torch.autograd import Variable
from load_memmap import *

class AxonDataset(Dataset):
    """" Inherits pytorch Dataset class to load Axon Dataset """
    def __init__(self, data_name='crops64_axons_only', folder='axon_data', type='train', transform=None, resize=None, normalise=False, read='npy'):
        """
        :param data_name (string)- data name to load/ save
        :param folder- location of dataset
        :param type - train or test dataset
        """
        self.data_name = data_name
        self.read = read
        self.transform = transform
        self.resize = resize
        self.normalise = normalise

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        if self.read == 'npy':
            self.x_data, self.y_data, _ = load_dataset(type, folder, data_name)
            self.len_data = len(self.x_data)
        elif self.read == 'image':
            self.folder = os.path.join(__location__,self.data_name,'train')
            images_original = [img for img in
                               os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.folder, "original"))]
            images_mask = [img for img in
                           os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.folder, "mask"))]
            self.images_mask = images_mask
            self.images_original = images_original
            self.images_mask.sort()
            self.images_original.sort()
            self.len_data = len(images_original)

    def __len__(self):
        """ get length of data
        example: len(data) """
        return self.len_data

    def __getitem__(self, idx):
        """gets samples from data according to idx
        :param idx- index to take
        example: data[10] -to get the 10th data sample"""
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        if self.read == 'npy':
            if self.resize:
                sample_x_data = np.resize(np.array([self.x_data[idx]]), (1, self.resize,self.resize))
                sample_y_data = np.resize(np.array([self.y_data[idx]]), (1, self.resize,self.resize))
            else:
                sample_x_data = self.x_data[idx]
                sample_y_data = self.y_data[idx]
        elif self.read == 'image':
            data_path = self.images_original[idx]
            mask_path = self.images_mask[idx]
            sample_x_data = plt.imread(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), self.folder, "original", data_path))
            sample_y_data = (plt.imread(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), self.folder, "mask", mask_path))).astype(
                float)
        sample_x_data = torch.Tensor(sample_x_data)
        sample_y_data = torch.Tensor(sample_y_data)

        if len(sample_x_data.shape) == 2:
            sample_x_data.unsqueeze_(0)
        if len(sample_y_data.shape) == 2:
            sample_y_data.unsqueeze_(0)

        # normalise between [-1,1]
        if self.normalise:
            sample_x_data = 2*((sample_x_data - torch.min(sample_x_data))/ (torch.max(sample_x_data) - torch.min(sample_x_data)) ) - 1

        data = [sample_x_data, sample_y_data]

        return data

class SyntheticDataset(Dataset):
    """" Inherits pytorch Dataset class to load Synthetic Axon Dataset """
    def __init__(self, num=50000, data_name='syn256', type='val', transform=None, resize=None):
        """
        :param num - number of data to generate
        :param data_name (string)- data name to load/ save
        :param type - train or test dataset
        """
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))
        name_x = os.path.join(__location__, 'npy_data/' + data_name + '_x_data_' + type + '.npy')
        name_y = os.path.join(__location__,'npy_data/' + data_name + '_y_data_' + type + '.npy')
        name_y_points = os.path.join(__location__,'npy_data/' + data_name + '_y_points_data_' + type + '.npy')

        try:
            self.x_data = np.load(name_x, mmap_mode='r')
            self.y_data = np.load(name_y, mmap_mode='r')
            self.y_data_points = np.load(name_y_points)
        except:
            # if no dataset currently created, generate a new synthetic dataset with parameters args
            print('no dataset with the name')
        self.data_name = data_name
        self.transform = transform
        self.resize = resize


    def read_tensor_dataset(self):
        """ converts dataset to tensors """
        tt = ToTensor()
        x_data = tt(self.x_data)
        y_data = tt(self.y_data)
        return x_data, y_data

    def __len__(self):
        """ get length of data
        example: len(data) """
        return (len(self.x_data))

    def __getitem__(self, idx):
        """gets samples from data according to idx
        :param idx- index to take
        example: data[10] -to get the 10th data sample"""

        if self.resize:
            sample_x_data = np.resize(np.array([self.x_data[idx]]), (1, self.resize,self.resize))
            sample_y_data = np.resize(np.array([self.y_data[idx]]), (1, self.resize,self.resize))
        else:
            sample_x_data = self.x_data[idx]
            sample_y_data = self.y_data[idx]
            sample_x_data = np.expand_dims(sample_x_data, axis=0)
            sample_y_data = np.expand_dims(sample_y_data, axis=0)

        sample_x_data = torch.Tensor(sample_x_data)
        sample_y_data = torch.Tensor(sample_y_data)


        data = [sample_x_data, sample_y_data]

        return data

class ToTensor:
    """Convert ndarrays in data to Tensors."""

    @staticmethod
    def __call__(data):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #data = data.transpose((1, 0))

        data = np.array([data])
        data = torch.Tensor(data)
        if torch.cuda.is_available():
            data = data.cuda()

        return data

    @staticmethod
    def data_to_tensor(x_data, y_data):
        """takes data and splits into a list of tensors- of which each list contains
        tensors of several samples (i.e. one id)
        :param x_data - the data
        :param y_data - the labels
        """

        tt = ToTensor()
        x_train_temp = tt(x_data)
        y_train_temp = tt(y_data)
        data = [x_train_temp, y_train_temp]
        return data


    @staticmethod
    def data_ids_to_tensor_list(x_data, y_data, ids):
        """takes data and splits into a list of tensors- of which each list contains
        tensors of several samples (i.e. one id)
        :param x_data - the data
        :param y_data - the labels
        :param ids - the ids corresponding to each sample
        """

        tt = ToTensor()
        unique_ids = np.unique(ids)
        data = [None] * unique_ids.size
        len = np.zeros(unique_ids.size).astype(int)
        for i in np.arange(unique_ids.size):
            ind_id = np.nonzero(unique_ids[i] == ids)[0].astype(int)
            len[i] = int(ind_id.size)
            x_train_temp = tt(x_data[ind_id])
            y_train_temp = tt(y_data[ind_id])
            data[i] = [x_train_temp[0], y_train_temp[0], len[i]]
        max_len = int(np.max(len))
        return data, max_len

    @staticmethod
    def create_variable(tensor):
        """creates a Variable tensor with gpu if available
        :param tensor - the tensor to wrap with Variable """

        # Do cuda() before wrapping with variable
        if torch.cuda.is_available():
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)
