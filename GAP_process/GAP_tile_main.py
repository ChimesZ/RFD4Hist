import pandas as pd
import os
from GAP_tile import label_segment, patch_annot

size = 224
path = '/home/zhong/data/GAP/'
label_path = path + 'Label/'
wsi_path = path + 'WSI/'
patch_path = path + f'Patch_{size}/'
annot_save_path = path + 'info/'

def main():
    info = pd.read_csv(path + 'GAP_info.csv')
    for i in range(len(info)):
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