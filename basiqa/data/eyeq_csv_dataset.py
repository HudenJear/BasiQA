import pandas as pd
import os
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from utils import FileClient, imfrombytes, img2tensor
from utils.registry import DATASET_REGISTRY
from .transforms import augment, center_crop
import numpy as np


@DATASET_REGISTRY.register()
class EyeQcsvDataset(data.Dataset):
    """EyeQ dataset for image quality assessment with csv sheet.
    Read image and its label(score) pairs.
    This method pass the index of label but not the one hot matrix.

    There is 1 mode:
    single images with a individual name + a csv file with all images` label.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their score.
            io_backend (dict): IO backend type and other kwarg.(disk only for this dataset yet)
            image_size (tuple): Resize the image into a fin size (should be square).
            suffix (str): in case of the suffix changed in the preprocessing, you can choose the suffix of the images(.jpeg for example).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(EyeQcsvDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.suffix=opt['suffix']

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])
        pd_data = pd.DataFrame(raw_data)
        # image,quality&DR_grade would be transfer to one hot matrix
        self.image_names = pd_data['image'].tolist()
        qualities = pd_data['quality'].to_numpy()
        DR_grade=pd_data['DR_grade'].to_numpy()
        # self.quals=ndarray2onehot(qualities)
        # self.DR_Gs=ndarray2onehot(DR_grade)
        self.quals=qualities
        self.DR_Gs=DR_grade

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        full_name,suffix0=os.path.splitext(self.image_names[index])
        img_path = os.path.join(self.dt_folder, full_name+self.suffix)
        img_data = cv2.imread(img_path)

        # augment and cut edge
        if self.opt['square_cut']:
            img_data=center_crop(img_data)
        if self.opt['flip']:
            img_data=augment(img_data)

        # resize image for train (val)
        image_size = (self.opt['image_size'], self.opt['image_size'])
        img_data = cv2.resize(img_data, image_size)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_data = img2tensor(img_data, bgr2rgb=True, float32=True)
        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(img_data, self.mean, self.std, inplace=True)

        return {'image': img_data, 'quality': self.quals[index], 'DR_grade': self.DR_Gs[index],'img_path': img_path}

    def __len__(self):
        return len(self.quals)


# def ndarray2onehot(ndar,num_class):
#   ar_len=len(ndar)
#   matr=np.zeros([ar_len,num_class])
#   for ind in range(0,ar_len):
#     matr[ind, ndar[ind]]=1

#   return matr
