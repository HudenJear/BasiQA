import pandas as pd
import numpy as np
import os
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from utils import FileClient, imfrombytes, img2tensor
from utils.registry import DATASET_REGISTRY
from .transforms import augment3
from PIL import Image


@DATASET_REGISTRY.register()
class ScoreImageDataset3(data.Dataset):
    """Single image dataset for image quality assessment.

    Read image and its label(score) pairs.

    There is 1 mode:
    single images with a individual name + a csv file with all images` label.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            image_folder (str): the folder containing all the images.
            csv_path (str): the csv file consists of all image names and their score.
            full_score (int/float): the full marks of images, 100 commonly.
            io_backend (dict): IO backend type and other kwarg.(disk only for this dataset yet)
            image_size (tuple): Resize the image into a fin size (should be square).

            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ScoreImageDataset3, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.resize = opt['resize'] if 'reszie' in opt else True
        if self.mean is not None or self.std is not None:
          print('Normlizing Active')
        self.augment_ratio=opt['augment_ratio'] if 'augment_ratio' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        csv_path = opt['csv_path']
        _, ext = os.path.splitext(csv_path)
        ext = ext.lower()
        if ext in ['.xlsx', '.xls']:
            engine = 'openpyxl' if ext == '.xlsx' else 'xlrd'
            raw_data = pd.read_excel(csv_path, engine=engine)
        else:
            raw_data = pd.read_csv(csv_path)
        pd_data = pd.DataFrame(raw_data)
        self.image_names = pd_data['file_name'].tolist()
        if 'meanOpinionScore' in pd_data.columns:
            self.scores = pd_data['meanOpinionScore'].tolist()
        else:
            self.scores = [None] * len(self.image_names)

        # make augment
        # directly increase the lenth of list, will effect the validation time and epochs counting
        if self.augment_ratio is not None:
          new_image_list=[]
          new_score_list=[]
          for times in range(0,self.augment_ratio):
            new_image_list.extend(self.image_names)
            new_score_list.extend(self.scores)
          self.image_names=new_image_list
          self.scores=new_score_list

        # make the score in numpy format and between (0-1)
        if all(v is not None for v in self.scores):
            self.scores = np.array(self.scores)
            self.scores = self.scores / opt['full_score']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def __getitem__(self, index):

        img_path = os.path.join(self.dt_folder, self.image_names[index])
        img_data = Image.open(img_path)
        score = self.scores[index]
        if score is None:
            score = float('nan')

        # augment
        # auto reshape and crop into size
        img_data=augment3(img_data.convert('RGB'),resize=self.resize, flip=self.opt['flip'],patch_size=self.opt['image_size'])
        # normalize (not recommanded)
        if self.mean is not None or self.std is not None:
            normalize(img_data, self.mean, self.std, inplace=True)

        return {'image': img_data, 'score': score, 'img_path': img_path}

    def __len__(self):
        return len(self.scores)


