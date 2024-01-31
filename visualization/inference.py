import openslide 
import os 
import numpy as np
from models import model_dict
import openslide 
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from utils import get_wsi_image, tessellation_tif, tessellation_wsi

def get_model_name(model_path):
    """parse teacher name"""
    if model_path.split('/')[-2].startswith('S:'):
        return model_path.split('/')[-2].split('_')[0].split(':')[-1]
    else:
        segments = model_path.split('/')[-2].split('_')
        if segments[0] != 'wrn':
            return segments[0]
        else:
            return segments[0] + '_' + segments[1] + '_' + segments[2]
def load_model(model_path, n_cls, model_name = None):
    print('==> loading teacher model')
    if model_name is None:
        model_t = get_model_name(model_path)
    else: 
        model_t = model_name
    model = model_dict[model_t](num_classes=n_cls)
    
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'], strict=False)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path, map_location=torch.device('cpu'))['model'].items()}) #Because parallel was used in training
    # model = torch.load(model_path)
    print('==> done')
    return model

class wsi_loader_infer(Dataset): 
    def __init__(self, spatial: list, wsi_path: str, tile_size: int = 150, transform = None) -> None:
        self.wsi = openslide.OpenSlide(wsi_path)
        wsi_name = wsi_path.split('/')[-1]
        self.center_list = spatial
        self.tile_size = tile_size
        self.transform = transform
    
    def __len__(self):
        return self.center_list.shape[0]

    def __getitem__(self, index):
        i, j = self.center_list[index][0], self.center_list[index][1]
        tile = self.wsi.read_region((int(i - self.tile_size/2), int(j - self.tile_size/2)), 0, (self.tile_size, self.tile_size))
        tile = np.asarray(tile, dtype=np.float32)[:, :, :3]
        if self.transform is not None: 
            tile = self.transform(image=tile)["image"]
        return tile, index

def wsi_get_spatial_dataloaders(spatial, wsi_path, tile_size: int = 150, batch_size=128, num_workers=8): 
    transform = A.Compose(
        [          
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Resize(150,150),
            ToTensorV2(),
        ]
    )
    dataset = wsi_loader_infer(spatial, wsi_path, tile_size=tile_size, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return dataloader


class img_loader_infer(Dataset): 
    def __init__(self, spatial: list, img: np.array, tile_size: int = 150, transform = None) -> None:
        self.img = img
        self.center_list = spatial
        self.tile_size = tile_size
        self.transform = transform
    
    def __len__(self):
        return self.center_list.shape[0]

    def __getitem__(self, index):
        i, j = self.center_list[index][1], self.center_list[index][0]
        tile = self.img[int(i - self.tile_size/2): int(i + self.tile_size/2), int(j - self.tile_size/2):int(j + self.tile_size/2)]
        tile = np.asarray(tile, dtype=np.float32)[:, :, :3]
        if self.transform is not None: 
            tile = self.transform(image=tile)["image"]
        return tile, index

def img_get_spatial_dataloaders(spatial, img, tile_size: int = 150, batch_size=128, num_workers=8): 
    transform = A.Compose(
        [          
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Resize(150,150),
            ToTensorV2(),
        ]
    )
    dataset = img_loader_infer(spatial, img, tile_size=tile_size, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    return dataloader

def inference_wsi(dataloader, model, device) -> torch.Tensor:
    model.eval()
    output_all = torch.tensor([]).to(device)
    with torch.no_grad():
        for input, _ in tqdm(dataloader):
            input = input.to(device)
            output = model(input)
            softmax = torch.nn.Softmax(dim=1)
            output = softmax(output)
            output_all = torch.cat((output_all, output))
    return output_all # Softmax

def process_img(img_path, model, device, tile_size=150, batch_size=128, num_workers=8) -> pd.DataFrame:
    wsi_im = cv2.imread(img_path)
    wsi_im = cv2.cvtColor(wsi_im, cv2.COLOR_BGR2RGB)
    spatial, _, _ = tessellation_tif(wsi_im, tile_size=tile_size)
    dataloader = img_get_spatial_dataloaders(spatial, wsi_im, tile_size=tile_size, batch_size=batch_size, num_workers=num_workers)
    output_all = inference_wsi(dataloader, model, device).cpu().numpy()
    df = pd.DataFrame(np.concatenate((spatial, output_all), axis=1), 
                      columns=['x', 'y', 'CTne', 'IT', 'LE', 'CT', 'CTmvp', 'CTpan', 'CTpnz', 'BG'])
    return df

def process_wsi(wsi_path, model_path, n_cls, device, model_name=None, tile_size=150, batch_size=128, num_workers=8) -> pd.DataFrame:
    spatial, _, _ = tessellation_wsi(wsi_path, tile_size=tile_size)
    model = load_model(model_path, n_cls, model_name)
    model.to(device)
    dataloader = wsi_get_spatial_dataloaders(spatial, wsi_path, tile_size=tile_size, batch_size=batch_size, num_workers=num_workers)
    output_all = inference_wsi(dataloader, model, device).cpu().numpy()
    df = pd.DataFrame(np.concatenate((spatial, output_all), axis=1), 
                      columns=['x', 'y', 'CTne', 'IT', 'LE', 'CT', 'CTmvp', 'CTpan', 'CTpnz', 'BG'])
    return df 

def main(): 
    img_path = '/data1/GAP/wsi/W4-1-1-A.1.03.jpg'
    img_name = img_path.split('/')[-1].remove('.jpg')
    model_path = '/home/lthpc/zhongzh/RFD4Hist/save/models/ResNet18_ivygap_lr_0.05_decay_0.0005_trial_0/ResNet18_best.pth'
    n_cls = 8
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(device)
    df = process_img(img_path, model_path, n_cls, device)
    df.to_csv(f'./{img_name}.csv', index=False)

if __name__ == '__main__': 
    pass