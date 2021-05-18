import os
import glob
import random

if __name__ == '__main__':
    train_data_path = '../data/train'
    val_data_path = './data/val/'
    labels = os.listdir(train_data_path)
    txt_path = '../data/'
    for index, label in enumerate(labels):
        if(label=='.DS_Store') :continue
        imglist = glob.glob('../data/train'+'/'+label+'/*')
        random.shuffle(imglist)
        with open(txt_path + 'train.txt', 'a')as f:
            for img in imglist:
                f.write((img + ' ' + label)[1:])
                f.write('\n')
    with open(txt_path + 'val.txt', 'a')as f:
        with open('../data/val_anno.txt', 'r') as t:
            for line in t.readlines():
                string = val_data_path + line
                if(len(string)>=12) : f.write(string)