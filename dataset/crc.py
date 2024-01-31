from __future__ import print_function
from __future__ import division
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset, DataLoader

def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if file_type in apath[-4:]:
                result.append(apath)
    return result


class CRC_Dataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        '''['NORM', 'MUC', 'TUM', 'BACK', 'MUS', 'LYM', 'DEB', 'STR', 'ADI']'''
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "NORM":
            label = 0
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "MUC":
            label = 1
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "TUM":
            label = 2
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "MUS":
            label = 3
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "LYM":
            label = 4
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "DEB":
            label = 5
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "STR":
            label = 6
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "ADI":
            label = 7
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "BACK":
            label = 8
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label, idx
    
def get_CRC_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    train_path = '/data1/CRC_K/NCT-CRC-HE-100K/'
    test_path = '/data1/CRC_K/CRC-VAL-HE-7K/'

    train_images_filepaths = all_path(train_path, '.tif')
    test_images_filepaths = all_path(test_path, '.tif')

    train_transform = A.Compose(
        [
            A.RandomRotate90(p = 0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
        
    train_set = CRC_Dataset(images_filepaths=train_images_filepaths, 
                                transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = CRC_Dataset(images_filepaths=test_images_filepaths, 
                                transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, len(train_set)
    else:
        return train_loader, test_loader

if __name__ == '__main__':
    pass 
