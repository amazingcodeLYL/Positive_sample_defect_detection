from torch.utils.data import DataLoader,Dataset
import PIL.Image as Image
import os
import cv2
from torchvision import transforms
import torch
import os
import torch
import imageio
from misc import imutils
import numpy as np

class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def make_tc_dataset(root):
    imgs = []

    naogeng = os.listdir(root)
    file_naogeng = []
    for i in range(len(naogeng)):
        file_naogeng_path = os.path.join(root, naogeng[i])
        file_naogeng.append(file_naogeng_path)
    for i in range(len(file_naogeng)):
        imgs.append({
            'name': naogeng[i],
            'img': file_naogeng[i],
            'label': torch.tensor([0, 1])
        })
    print(len(imgs))
    return imgs

class tc_dataset(Dataset):
    def __init__(self, root, resize_long=None, rescale=None, img_normal=TorchvisionNormalize(),
                 hor_flip=False, crop_size=None, crop_method=None, to_torch=True):
        imgs = make_tc_dataset(root)
        self.imgs = imgs
        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __getitem__(self, index):
        datafiles = self.imgs[index]
        name = datafiles['name']
        x_path = datafiles['img']
        img = np.asarray(imageio.imread(x_path))
        if img.shape != (128, 128, 3):
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img, img, img), axis=-1)

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])
        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)
        if self.img_normal:
            img = self.img_normal(img)
        if self.hor_flip:
            img = imutils.random_lr_flip(img)
        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)
        if self.to_torch:
            img = imutils.HWC_to_CHW(img)
        return {'name': name, 'img': img}

    def __len__(self):
        return len(self.imgs)

class tc_dataloader(Dataset):
    def __init__(self, root, scales=(1.0, ), img_normal=TorchvisionNormalize()):
        imgs = make_tc_dataset(root)
        self.imgs = imgs
        self.scales = scales
        self.img_normal = img_normal

    def __getitem__(self, index):
        datafiles = self.imgs[index]
        name = datafiles['name']
        x_path = datafiles['img']
        img_y = datafiles['label']
        img = imageio.imread(x_path)
        if img.shape != (128, 128, 3):
            img = np.expand_dims(img, axis=2)
            img = np.concatenate((img, img, img), axis=-1)
        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)
            s_img = self.img_normal(s_img)
            #print(s_img.shape)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))  # np.flip(img, -1)左右翻转, 将原图与翻转后的图像拼在一起
        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]
        out = {"name": name, "img": ms_img_list, "size": (img.shape[0], img.shape[1]),
               "label": img_y}
        return out

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    dataset = tc_dataset("../raw_data/round_test/part1/TC_Images")
    #dataset = tc_dataloader("../fusai_data/part1/TC_Images", scales=(1.0, 0.5, 1.5, 2.0))
    data_loader = DataLoader(dataset, shuffle=False, num_workers=4 , pin_memory=False)
    for iter, pack in enumerate(data_loader):
        pass
        # img_name = pack['name'][0]
        # label = pack['label'][0]
        # size = pack['size']
        # img = pack['img']
        # print(img_name)
        # print(label)
        # print(size)
        # print(len(img))  # 4  四种不同的尺寸，对应cam_scales中的四个
        # print(img[0].shape)  # (1,2,3,256,256)
        # print(img[1].shape)  # (1, 2, 3, 128, 128) 对应cam_scales = 0.5
        # print(img[2].shape)  # (1,2,3,384,384)
        # print(img[3].shape)  # (1,2,3,512,512)