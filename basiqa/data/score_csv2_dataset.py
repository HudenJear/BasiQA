import pandas as pd
import os
import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from utils import FileClient, imfrombytes, img2tensor
from utils.registry import DATASET_REGISTRY
from .transforms import augment2
from PIL import Image


@DATASET_REGISTRY.register()
class ScoreImageDataset2(data.Dataset):
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
        super(ScoreImageDataset2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']  # only disk type is prepared for this dataset type.
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # read csv and build a data list
        self.dt_folder = opt['image_folder']
        raw_data = pd.read_csv(opt['csv_path'])
        pd_data = pd.DataFrame(raw_data)
        self.image_names = pd_data['file_name'].tolist()
        self.scores = pd_data['opinion_score'].to_numpy()

        # make the score between (0-1)
        self.scores = self.scores / opt['full_score']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def __getitem__(self, index):

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_path = os.path.join(self.dt_folder, self.image_names[index])
        # print(img_path)
        img_data = Image.open(img_path)
        score = self.scores[index]

        # augment and cut edge
        if self.opt['flip']:
            img_data=augment2(img_data.convert('RGB'),flip=self.opt['flip'],patch_size=self.opt['image_size'])

        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_data = img2tensor(img_data, bgr2rgb=True, float32=True)
        # # normalize (not recommanded)
        # if self.mean is not None or self.std is not None:
        #     normalize(img_data, self.mean, self.std, inplace=True)

        return {'image': img_data, 'score': score, 'img_path': img_path}

    def __len__(self):
        return len(self.scores)


