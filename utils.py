import os
import torch
import torch.utils.data
import numpy as np
import sys
from skimage.feature import hog
from matplotlib import pyplot as plt
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


# Given a directory, iterate over all images and output feature vector and label vector (np.arrays)
def vectorize_and_label(directory, do_cuts=False, phi=0, do_hog=False, orientations=8, ppc=8, cpb=4, block_norm='L2'):
    image_vect = []
    label_vect = []
    for i_dir in directory:
        for filename in os.listdir(i_dir):
            label_vect.append(int(filename[6:8]))
            file_dir = i_dir+'/'+filename
            image = plt.imread(file_dir)
            if do_cuts and do_hog:
                print("Can't do both cuts and hog...")
                sys.exit()
            elif do_cuts:
                cuts = return_pattern_cuts(image,phi)
                image_vect.append(cuts)
            elif do_hog:
                hog_feature = hog(image, orientations=orientations, pixels_per_cell=(ppc,ppc), cells_per_block=(cpb,cpb), block_norm=block_norm)
                image_vect.append(hog_feature)
            else:
                image_vect.append(image.ravel())
    return np.array(image_vect), np.array(label_vect)
        

# Function to return azimuth/elevation cuts given far field pattern
# Only works when the signal is in the azimuth plane so far
def return_pattern_cuts(patt, phi):
    phi = -phi;
    phi_vec = np.arange(-90,90,181/patt.shape[1])
    phi_idx = np.argmin(np.abs(phi_vec-phi))
    el_cut = patt[:,phi_idx]
    az_cut = patt[int(patt.shape[0]/2),:]
    cuts = np.r_[el_cut,az_cut]
    return cuts


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def make_dataset (traindir):
    img = []
    for i_dir in traindir:
        for subdir, dirs, files in os.walk(i_dir):
            for fname in files:
                target = int(fname[6:8])
                path = os.path.join(subdir, fname)
                item = (path, target)
                img.append(item)        
    return img


class ArrayDataset (torch.utils.data.Dataset):
    def __init__(self, traindir, transform=None, target_transform =None,
                loader = pil_loader):
        self.traindir = traindir
        self.imgs = make_dataset (traindir)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)