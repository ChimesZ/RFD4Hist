import openslide 
import numpy as np
import openslide 
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



def tessellation_wsi(wsi: openslide.OpenSlide, level = 0, save_path = None, tile_size = 150): 
    array_row, array_col, spatial = [], [], []
    is_tissue = []
    m = 0
    for i in range(0, wsi.level_dimensions[level][0], int(tile_size)):
        n=0
        for j in range(0, wsi.level_dimensions[level][1], int(tile_size)):
            # tile = wsi.read_region((i, j), level, (int(tile_size), int(tile_size)))
            # if save_path is not None: 
            #     tile.save(os.path.join(save_path, f'{i+int(tile_size/2)}_{j+int(tile_size/2)}.png'))
            # tile = np.asarray(tile, dtype=np.float32)[:, :, :3]/255
            if i + int(tile_size) <= wsi.dimensions[0] and j + int(tile_size) <= wsi.dimensions[1]: 
                spatial.append([i+int(tile_size/2), j+int(tile_size/2)])
                array_row.append(n)
                array_col.append(m)
            n+=1
        m+=1
    return np.array(spatial, dtype=int), array_row, array_col

def tessellation_tif(img: np.array, save_path = None, tile_size = 150): 
    array_row, array_col, spatial = [], [], []
    is_tissue = []
    m = 0
    for i in range(0, img.shape[1], int(tile_size)):
        n=0
        for j in range(0, img.shape[0], int(tile_size)):
            # tile = wsi.read_region((i, j), level, (int(tile_size), int(tile_size)))
            # if save_path is not None: 
            #     tile.save(os.path.join(save_path, f'{i+int(tile_size/2)}_{j+int(tile_size/2)}.png'))
            # tile = np.asarray(tile, dtype=np.float32)[:, :, :3]/255
            if i + int(tile_size) <= img.shape[1] and j + int(tile_size) <= img.shape[0]: 
                spatial.append([i+int(tile_size/2), j+int(tile_size/2)])
                array_row.append(n)
                array_col.append(m)
            n+=1
        m+=1
    return np.array(spatial, dtype=int), array_row, array_col


def get_wsi_image(wsi, level) -> np.array:
    level_count = wsi.level_count
    assert level in range(level_count)
    level_dimensions = wsi.level_dimensions[level]
    img = wsi.read_region((0, 0), level, level_dimensions) # 此处返回为PIL.Image格式，转化为np.array
    img = np.asarray(img, dtype=np.float32)[:, :, :3]/255
    return img