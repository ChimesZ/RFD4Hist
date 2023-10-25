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


class GAP_Dataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "CTne":
            label = 0
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "IT":
            label = 1
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "LE":
            label = 2
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "CT":
            label = 3
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "CTmvp":
            label = 3
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "CTpan":
            label = 3
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "CTpnz":
            label = 3
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "BG":
            label = 4
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label
    
def get_GAP_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    cifar 100
    """
    train_path = '/home/zhong/data/GAP/Dataset/train/'
    test_path = '/home/zhong/data/GAP/Dataset/val/'

    train_images_filepaths = all_path(train_path, '.jpg')
    test_images_filepaths = all_path(test_path, '.jpg')

    train_transform = A.Compose(
        [
            # # A.SmallestMaxSize(max_size=160),
            # # A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=45, p=1),
            # # A.RandomCrop(height=128, width=128),
            # A.Blur(blur_limit=3),
            A.RandomRotate90(p = 0.5),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
            # A.RandomBrightnessContrast(p=1),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            # A.SmallestMaxSize(max_size=160),
            # A.CenterCrop(height=128, width=128),
            # A.RandomRotate90(p = 0.5),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
            # A.RandomBrightnessContrast(p=1),
            # A.VerticalFlip(p=0.5),
            # A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
        
    #####OPERATION#####

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # std=[0.229, 0.224, 0.225])
    # train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
        # normalize,
    # ])
    # test_transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        # normalize,
    # ])
    #####END#####

    if is_instance:
        pass
    else:
        train_set = GAP_Dataset(images_filepaths=train_images_filepaths, 
                                transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = GAP_Dataset(images_filepaths=test_images_filepaths, 
                                transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        pass
    else:
        return train_loader, test_loader



# train_transform = A.Compose(
#     [
#         # # A.SmallestMaxSize(max_size=160),
#         # # A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=45, p=1),
#         # # A.RandomCrop(height=128, width=128),
#         # A.Blur(blur_limit=3),
#         A.RandomRotate90(p = 0.5),
#         # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
#         # A.RandomBrightnessContrast(p=1),
#         A.VerticalFlip(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )

# val_transform = A.Compose(
#     [
#         # A.SmallestMaxSize(max_size=160),
#         # A.CenterCrop(height=128, width=128),
#         # A.RandomRotate90(p = 0.5),
#         # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1),
#         # A.RandomBrightnessContrast(p=1),
#         # A.VerticalFlip(p=0.5),
#         # A.HorizontalFlip(p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )