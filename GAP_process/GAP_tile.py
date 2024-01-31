import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os



def label_segment(filepath, filename, savepath='', size=150): 
    """Generate location of each type of WSI patch

    Args:
        filepath (str): path to label picture
        filename (str): name of label picture
        savepath (str, optional): path to save of segmentation information annot. Defaults to ''.
        size (int, optional): size of each patch. Defaults to 150.

    Returns:
        pd.Dataframe: information to guide patch annotation
    """

    key = ['CTne','IT','LE','CT','CTmvp','CTpan','CTpnz','BG']

    arrange = {
        'type': key,
        'gray_value': [5, 89, 113, 124, 136, 143, 171, 255],
        'position': [[] for _ in range(len(key))]
    }
    label = cv2.imread(filepath + filename + '.jpg')
    label_gray = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
    annot = pd.DataFrame(arrange)
    h,w = label_gray.shape
    wn = int(np.floor(w/size))
    hn = int(np.floor(h/size))
    for i in tqdm(range(hn)):
        for j in range(wn):
            patch = label_gray[size*i:size*(i+1),size*j:size*(j+1)]
            mean = patch.mean().mean()
            for k in range(len(annot)):
                if mean == annot['gray_value'][k]:
                    annot['position'][k].append((size*i,size*j))
    annot['num'] = [len(annot['position'][i]) for i in range(len(annot))]
    if savepath != '': 
        annot.to_csv(savepath + filename + '_annot.csv')
    return annot

def patch_annot(filepath, filename, savepath, annot:pd.DataFrame, size=150):
    # TODO Finish the doc
    """_summary_

    Args:
        filepath (_type_): _description_
        filename (_type_): _description_
        savepath (_type_): _description_
        annot (pd.DataFrame): _description_
        size (int, optional): _description_. Defaults to 150.
    """
    wsi = cv2.imread(filepath + filename + '.jpg')
    # wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)
    for i in range(len(annot)):
        typepath = savepath + annot['type'][i] + '/'
        if os.path.exists(typepath) is not True:
            os.mkdir(typepath)
        position = annot['position'][i]
        for h, w in position:
            patch = wsi[h:h+size, w:w+size, :]
            # np.save(typepath + filename + f'_({h},{w}).npy',patch)
            if patch.shape == (size,size,3):
                cv2.imwrite(typepath + filename + f'_({h},{w}).jpg',
                            patch,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        print('Number of {} patches: {}'.format(annot['type'][i],
                                                annot['num'][i]))


def main():
    size = 150
    path = '/data1/GAP/'
    label_path = path + 'Label/'
    wsi_path = path + 'wsi/'
    patch_path = path + f'Patch_{size}/'
    if not os.path.exists(patch_path): os.mkdir(patch_path)
    annot_save_path = path + 'info/'
    if not os.path.exists(annot_save_path): os.mkdir(annot_save_path)
    info = pd.read_csv(path + 'GAP_info.csv')
    for i in tqdm(range(len(info))):
        tag = info['tag'][i]
        tag_path = patch_path + tag + '/'
        if os.path.exists(tag_path) is False: 
            os.mkdir(tag_path)
        names = eval(info['file'][i])
        for name in names: 
            print(f'===Begin to process segmentation picture of {name}===')
            annot = label_segment(filepath=label_path, 
                            filename=name+'_L', 
                            savepath = annot_save_path,
                            size=size)
            print(f'Begin to process WSI {name}')
            patch_annot(filepath=wsi_path, 
                        filename=name, 
                        savepath=tag_path,
                        annot=annot,
                        size=size)
            print('===========Finished===========')

if __name__ == '__main__': 
    main()
# if __name__ == '__main__': 
#     filepath = '/home/ChimesZ/data/pathology/'
#     filename_L = 'W10-1-1-B.2.02_L'
#     filename_W = 'W10-1-1-B.2.02'
#     savepath = '/home/ChimesZ/data/pathology/seg_test/'

#     print(f'Begin to process segmentation picture of {filename_W}')

#     annot = label_segment(filepath=filepath, 
#                           filename=filename_L, 
#                           savepath = savepath)

#     print(f'Finished!')

#     print(f'Begin to process WSI {filename_W}')

#     patch_annot(filepath=filepath, 
#                 filename=filename_W, 
#                 savepath=savepath,
#                 annot=annot)

#     print('Finished!')


