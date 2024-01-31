import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.patches import Rectangle
import os 
import openslide
from tqdm import tqdm

# COLOR_DICT = {'CTne': 'black',
#  'IT': 'pink',
#  'LE': 'blue',
#  'CT': 'purple',
#  'CTmvp': 'red',
#  'CTpan': 'orange',
#  'CTpnz': 'gray',
#  'BG': 'white'}

COLOR_DICT = {
    'CTne': 'black',
    'IT': '#AF309A',
    'LE': '#0090A7',
    'CT': '#19B957',
    'CTmvp': '#FF4414',
    'CTpan': '#00BC96',
    'CTpnz': '#20CAEA',
    'BG': 'white'
    }

class img_generator():
    
    def __init__(self, info: pd.DataFrame, size = 150, downsample = 16) -> None:
        self.x = (np.array(info['x']) - size/2)
        self.y = (np.array(info['y']) - size/2)
        self.prob = np.array(info.iloc[:, 2:])
        self.type = info.iloc[:, 2:].idxmax(axis=1)
        self.size = size 
        self.downsample = downsample
        self.shape = (int(np.max(self.x)/size), int(np.max(self.y)/size))

    def len(self): 
        return len(self.x)

    def generate_patch_img(self):
        for i in range(len(self.x)): 
            yield Rectangle(
                (self.x[i]/self.downsample, self.y[i]/self.downsample), 
                self.size/self.downsample,
                self.size/self.downsample, 
                color = COLOR_DICT[self.type[i]], 
                alpha = 0.5
            )

    def generate_patch_sketch(self,patch_size):
        x = self.x / self.size
        y = self.y / self.size
        for i in range(len(x)):
            yield Rectangle(
                (x[i]*patch_size, y[i]*patch_size), 
                patch_size,
                patch_size, 
                color = COLOR_DICT[self.type[i]], 
                alpha = 1
            ) 
    
    def generate_channels(self): 
        im = np.zeros((self.shape[0],self.shape[1], 8))
        for i in range(len(self.x)):
            im[int(self.y[i]/self.size), int(self.x[i]/self.size)] = self.prob[i]
        return im 
    
    def generate_type(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'type': self.type,
        })
        return df
    

    def segmentation(self, save = None):
        fig, ax = plt.subplots()
        ax.imshow(np.ones((self.shape[1]*10, self.shape[0]*10)))
        # patch_gen = img_generator(df)
        for rec in tqdm(self.generate_patch_sketch(10)):
            ax.add_patch(rec)
        if save is not None:
            plt.savefig(os.path.join(save))
        return fig, ax



def main():
    df = pd.read_csv('/home/lthpc/zhongzh/RFD4Hist/visualization/W4-1-1-A.1.03.jpg.csv')
    classes = list(df.columns)[2:]
    patch_gen = img_generator(df)
    img = patch_gen.generate_channels()
    fig, axes = plt.subplots(2,4, figsize = (20,15))
    for i, type in enumerate(classes):
        axes[int(np.floor(i/4)), int(i%4)].imshow(img[:,:,i])
        # axes[int(np.floor(i/4)), int(i%4)].title(type)
        axes[int(np.floor(i/4)), int(i%4)].set_title(type) 
    plt.show()

if __name__ == '__main__':
    patch_gen = img_generator(pd.read_csv('/home/lthpc/zhongzh/RFD4Hist/visualization/W4-1-1-A.1.03.jpg.csv'))
    patch_gen.segmentation('/home/lthpc/zhongzh/RFD4Hist/visualization/figrues/test.svg')

