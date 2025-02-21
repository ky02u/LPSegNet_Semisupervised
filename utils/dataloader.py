import os 
from PIL import Image

import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd

class PolypData(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize, cls_, cls_path):
   
        # self.images_root = sorted([image_root + file for file in os.listdir(image_root) if file.endswith('.jpg') or file.endswith('.png')])
        # self.gts_root = sorted([gt_root + file for file in os.listdir(gt_root) if file.endswith('.jpg') or file.endswith('.png')])
        self.images_root = image_root
        self.gts_root = gt_root
        self.trainsize = trainsize
        self.cls_ = cls_
        self.cls_path = cls_path
        
        self.filter_files()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        if self.cls_:
            self.polyp_image, self.polyp_label = self.read_csv(cls_path)
    def __len__(self):
        return len(self.images_root)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images_root[index])
        image = self.img_transform(image)

        gt = self.binary_loader(self.gts_root[index])
        gt = self.gt_transform(gt)
    
        if self.cls_:
            name = 1
            name_img = self.images_root[index].split('/')[-1]
            index_image = np.where(int(name_img[:-4]) == self.polyp_image)[0][0]
            label_histhology =  self.polyp_label[index_image]
            label = int(name)
        else: 
            name = self.images_root[index].split('-')[0]
            name_img = self.images_root[index].split('/')[-1]
            label = int(name[-1])
            label_histhology = 0

        # labels = []

        # if label == 0: 
        #     labels.append([[[1]], [[0]]])
        # else: 
        #     labels.append([[[1]], [[1]]])

        # array = np.array(labels)
        # labels = torch.tensor(array)

        # label = label_.unsqueeze(1).unsqueeze(2).unsqueeze(3) #with one label
        
        # return image, gt, label, name_img, label_histhology
        return image, gt, label, name_img

    def read_csv(self, path):
        data_label = pd.read_csv(path)  
        image = data_label['image_id'].to_numpy()
        label = data_label['Histologia'].to_numpy()
        return image, label 

    def filter_files(self):
        assert len(self.images_root) == len(self.gts_root)
        images, gts = [], []
        for img_path, gt_path in zip(self.images_root, self.gts_root):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    path = '/home/linamruiz/Polyps/tesismaestria/LPSegNet/dataset_cls/data/m_train/train_2.csv'
    dataset = PolypData(image_root, gt_root, trainsize, False, path)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize, cls_, cls_path):
        self.testsize = testsize
        self.cls_ = cls_
        self.cls_path = cls_path
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        assert len(self.images) == len(self.gts)
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

        if self.cls_:
            self.polyp_image, self.polyp_label = self.read_csv(cls_path)

    def read_csv(self, path):
        data_label = pd.read_csv(path)  
        image = data_label['image_id'].to_numpy()
        label = data_label['Histologia'].to_numpy()
        return image, label 

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name_img = self.images[self.index].split('/')[-1]
        if name_img.endswith('.jpg'):
            name_img = name_img.split('.jpg')[0] + '.png'
        name = 1
        if self.cls_:
            index_image = np.where(int(name_img[:-4]) == self.polyp_image)[0][0]
            label_histhology =  self.polyp_label[index_image]
        else: label_histhology = 0
        self.index += 1

        return image, gt, name_img, name, label_histhology

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

# """
# Test the PolypData class 
# """
# image_root = "/home/linamruiz/Documentos/Tesis/data/TrainDataset/image/"
# gt_root = "/home/linamruiz/Documentos/Tesis/data/TrainDataset/mask/"

# data_loader = get_loader(image_root, gt_root, trainsize=352, batchsize=2)

# for i, data_load in enumerate(data_loader):
#     images, gts = data_load
#     images = Variable(images).cuda()
#     gts = Variable(gts).cuda()
#     print(images.size())
#     print('--------------------------------------------------')
#     break