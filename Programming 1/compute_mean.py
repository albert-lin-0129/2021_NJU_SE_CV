import numpy as np
import cv2
import os
import torch
import torchvision

img_h, img_w = 64, 64  # 根据自己数据集适当调整，影响不大
means = [0, 0, 0]
stdevs = [0, 0, 0]
img_list = []
label_path = './data/train'
label_path_list = os.listdir(label_path)
print(label_path_list)
num_imgs = 0
for label in label_path_list:
    if(label==".DS_Store"): continue
    imgs_path = os.path.join(label_path, label)
    imgs_list = os.listdir(imgs_path)
    for data in imgs_list:
        print(data)
        num_imgs += 1
        img = cv2.imread(os.path.join(imgs_path, data))
        img = img.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
means.reverse()
stdevs.reverse()
means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))