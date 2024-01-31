from __future__ import division
import stainNorm_Macenko
import stain_utils as utils
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

def all_path(dirname, file_type):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            if file_type in apath[-4:]:
                result.append(apath)
    return result


def norm(i1, input_path, save_base_dir):
    save_dir = os.path.join(save_base_dir, input_path.split('/')[-3], input_path.split('/')[-2])
    # Might raise error in multi-processing
    try:
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
    except:
        pass
    save_path = os.path.join(save_dir, input_path.split('/')[-1])
    i2=utils.read_image(input_path)#input
    try:
        n=stainNorm_Macenko.Normalizer()
        n.fit(i1)
        normalized=n.transform(i2)
    except:
        normalized=i2
    # cv.imwrite(save_dir, cv.cvtColor(normalized, cv.COLOR_RGB2BGR))
    im = Image.fromarray(normalized)
    im.save(save_path)
    
    return im


# path = r"/home/zhong/data/SptialOmicGBM_Patches" 
# # path = r"/media/aa/HDD/PathFinder4COAD/code/Stain_Normalization/new_data"
# paths = all_path(path,'.jpg')
# paths.sort()

# i1 = utils.read_image('/home/zhong/Experiment/AIGBM/PathFinder4GBM/code/Stain_Normalization/TCGA-S9-A89Z-01Z-00-DX1.jpg')
# # input_path = '/media/aa/HDD/PathFinder4COAD/code/Stain_Normalization/new_data/i4.jpg'
# save_base_dir = '/home/zhong/data/SptialOmicGBM_Patches_Norm'


# for i in tqdm(paths):
#     norm(i1, i, save_base_dir)


# num_processes = multiprocessing.cpu_count()
# print(num_processes)
# proccess_number = num_processes-2
# executor = ProcessPoolExecutor(max_workers=30)
# # task_list = [executor.submit(generate_multiclasses, matrix_dir, save_path) for matrix_dir in matrix_paths[140:210]]
# task_list = [executor.submit(norm, i1, matrix_dir, save_base_dir) for matrix_dir in paths]
# executor.shutdown(wait=True)

def get_args():
    parser = argparse.ArgumentParser(description='Train prognosis model for semantic segmented GBM')
    parser.add_argument('--data_path',
                        type=str,
                        default = '/data1/GAP/Dataset_10000_5')
    parser.add_argument('--save_path', 
                        type=str, 
                        default='/data1/GAP/Dataset_10000_5_Norm',
                        help='Dictionary to save experiment results')
    parser.add_argument('--ref_path',
                        type=str,
                        help='ref image',
                        default='/home/lthpc/zhongzh/RFD4Hist/Stain_Normalization/TCGA-S9-A89Z-01Z-00-DX1.jpg')
    parser.add_argument('--worker',
                        type=int,
                        default=None,
                        help='Number of workers')
    return parser

def main():
    args = get_args().parse_args()
    path = args.data_path
# path = r"/media/aa/HDD/PathFinder4COAD/code/Stain_Normalization/new_data"
    paths = all_path(path,'.jpg')
    paths.sort()

    i1 = utils.read_image(args.ref_path)
    # input_path = '/media/aa/HDD/PathFinder4COAD/code/Stain_Normalization/new_data/i4.jpg'
    save_base_dir = args.save_path
    num_processes = multiprocessing.cpu_count()
    if args.worker is not None:
        proccess_number = args.worker
        print(args.worker)
    else: 
        print(num_processes)
        proccess_number = num_processes-10
    executor = ProcessPoolExecutor(max_workers=proccess_number)
    processes = [executor.submit(norm, i1, matrix_dir, save_base_dir) for matrix_dir in paths]
    with tqdm(total=len(paths)) as pbar:
        for future in as_completed(processes):
            _ = future.result()
            pbar.update(1)
    executor.shutdown(wait=True)

if __name__ == '__main__':
    main()