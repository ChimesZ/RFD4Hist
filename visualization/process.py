from visualization import img_generator
from inference import process_img, process_wsi, load_model

import os 
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from random import sample
from tqdm import tqdm

def get_img_path_list(root_img, root_mask, num):
    file_list = os.listdir(root_img)
    path_img_list = [os.path.join(root_img, file) for file in file_list]
    sel_path_img_list = sample(path_img_list, num)
    sel_path_mask_list = [path.replace('.jpg', '_L.jpg') for path in sel_path_img_list]
    return sel_path_img_list, sel_path_mask_list


def main():
    root_img = '/data1/GAP/wsi'
    root_mask = '/data1/GAP/Label'
    save_root = '/home/lthpc/zhongzh/RFD4Hist/visualization/inference/'
    teacher = 'resnet32x4' 
    student = 'MobileNetV2'
    model_path_list = [
                       f'/home/lthpc/zhongzh/RFD4Hist/save/models/{student}_ivygap_lr_0.005_decay_0.0005_trial_0/{student}_best.pth', 
                       f'/home/lthpc/zhongzh/RFD4Hist/save/models/{teacher}_ivygap_lr_0.005_decay_0.0005_trial_0/{teacher}_best.pth',
                       f'/home/lthpc/zhongzh/RFD4Hist/save/student_model/S:{student}_T:{teacher}_ivygap_md_relation_pyr_r:1.0_a:0.0_b:0.7_1/ckpt_epoch_120.pth',
                       ]
    save_root = '/home/lthpc/zhongzh/RFD4Hist/save/inference'
    n_cls = 8
    path_img_list, path_mask_list = get_img_path_list(root_img, root_mask, 10)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    for model_path in model_path_list:
        print(f'Processing {model_path}')
        model = load_model(model_path, n_cls)
        model.to(device)
        save_path = os.path.join(save_root, model_path.split('/')[-2])
        save_img_path = save_path.replace('inference', 'figrues')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)
        for img_path in path_img_list:
            img_name = img_path.split('/')[-1].replace('.jpg', '')
            print(f'Processing {img_name}')
            df = process_img(img_path, model, device, batch_size=128)
            df.to_csv(os.path.join(save_path, f'{img_name}.csv'), index=False)
            print(f'Saved {img_name}.csv')
            patch_gen = img_generator(df)
            fig, ax = patch_gen.segmentation(os.path.join(save_img_path, f'{img_name}.svg'))
            print(f'Saved {img_name}.svg')

if __name__ == '__main__':
    main()

