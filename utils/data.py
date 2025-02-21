import os
from PIL import Image
import numpy as np
import cv2

import torch.utils.data as data
from sklearn.model_selection import train_test_split

"""
Load all data in a new folder call as data. 
"""
class PolypData(data.Dataset):
    def __init__(self, image_root, gt_root, fp_root, trainsize):
        
        os.makedirs('./dataset_IGHO_2/images_/', exist_ok=True)
        os.makedirs('./dataset_IGHO_2/masks_/', exist_ok=True)
        
        # self.fp_root  = []
        # for fp_videos in os.listdir(fp_root): 
        #     frames = os.listdir(fp_root + fp_videos + '/frames/')
        #     choice = np.random.choice(frames, 290, replace=False)
        #     for file in choice: 
        #         if file.endswith('.jpg') or file.endswith('.png'):
        #             self.fp_root.append(fp_root + fp_videos + '/frames/' + file)
        self.images_root = sorted([image_root + file for file in os.listdir(image_root) if file.endswith('.jpg') or file.endswith('.png')])
        self.read_json()

        self.fp_root         = sorted(self.choice_no_polyp)
        # self.new_images_root = sorted(self.choice_polyp)
        # self.gts_root        = sorted([file.replace('Normal', 'MASKS') for file in self.choice_polyp])
        # self.gts_root = sorted([gt_root + file for file in os.listdir(gt_root) if file.endswith('.jpg') or file.endswith('.png')])
        
        # self.fp_root = [fp_root + file for file in os.listdir(fp_videos) if file.endswith('.jpg') or file.endswith('.png')]
        # self.fps_root = np.random.choice(self.fp_root, 1500)

        self.trainsize = trainsize
        
        self.filter_files_np()
        # self.filter_files_wp()

    def read_json(self): 
        json_root = sorted([image_root + file for file in os.listdir(image_root) if file.endswith('.json')])
        if len(json_root) > 0:
            polyp_root = [i.replace('json', 'png') for i in json_root]
            # self.choice_polyp = np.random.choice(polyp_root, 281, replace=False)
            self.choice_polyp = polyp_root
            # np_root =  [i for i in self.images_root if i not in polyp_root]
        else: 
            np_root =  [i for i in self.images_root]
            self.choice_no_polyp = np.random.choice(np_root, 290, replace=False)
        
    def filter_files_wp(self):
        assert len(self.new_images_root) == len(self.gts_root)
        # images, gts = [], []
        
        for img_path, gt_path in zip(self.new_images_root, self.gts_root):

            img = Image.open(img_path)
            gt = Image.open(gt_path)
            
            new_img_path = './dataset_IGHO_2/images_/'
            new_gt_path = './dataset_IGHO_2/masks_/'
            name_video = img_path.split('/')[-3][11:]
            name_img = '1-' + name_video + '-' +  img_path.split('/')[-1]

            img.save(new_img_path + name_img)
            gt.save(new_gt_path + name_img)

        #     if img.size == gt.size:
        #         images.append(new_img_path + name_img)
        #         gts.append(new_gt_path + name_img)
        # print(len(images))
        # self.images = images
        # self.gts = gts

    def filter_files_np(self):
        for i, fp_path in enumerate(sorted(self.fp_root)):
            img = Image.open(fp_path)
            # img_c = cv2.imread(fp_path)
            gt = np.zeros((img.size[1], img.size[0]))

            new_img_path = './dataset_IGHO_2/images/'
            new_gt_path = './dataset_IGHO_2/masks/'
            
            name_video = fp_path.split('/')[-3][11:]
            name_img = '0-' + name_video + '-' + fp_path.split('/')[-1]

            img.save(new_img_path + name_img)
            cv2.imwrite(new_gt_path + name_img, gt)
            
            # if img_c.shape[:-1] == gt.shape:
            #     images.append(new_img_path + name_img)
            #     gts.append(new_gt_path + name_img)

"""
Test the PolypData class 
"""

# image_root = "/data/Datasets/Igho/Uncompressed/Videos/2021_NOVIEMBRE/polipo/2021-11-25_175942_606/Normal/"
image_root = "/data/Datasets/Igho/Uncompressed/Videos/2021_AGOSTO/NP/Colonoscopy/Train/2021-08-23_105392/Normal/"
gt_root = "/data/Datasets/Igho/Uncompressed/Videos/2021_AGOSTO/NP/Colonoscopy/Train/2021-08-23_105392/Normal/"
fp_root = ""

# PolypData(image_root, gt_root, fp_root, trainsize=352)  ##COMENTAR


def generate_train_test_split(image_root, gt_root): 

    images_root = [image_root + file for file in os.listdir(image_root) if file.endswith('.jpg') or file.endswith('.png')]
    gts_root = [gt_root + file for file in os.listdir(gt_root) if file.endswith('.jpg') or file.endswith('.png')]

    count_polyps, count_background = 0, 0
    total = len(images_root)
    for i, file_ in enumerate(images_root): 
        polyp_class = int(file_.split('-')[0][-1])
        # polyp_class = 1
        if polyp_class == 1: 
            count_polyps += 1
        else: 
            count_background += 1
    
    train_wp, val_wp = int( count_polyps*0.75), int(np.ceil(count_polyps*0.25)) #HICE CAMBIOS AQUI
    train_np, val_np = int(count_background*0.75), int(np.ceil(count_background*0.25))
    
    print(count_polyps, train_wp, val_wp, "BG: ", count_background, train_np, val_np)
    
    polyp_train_wp, polyp_val_wp, gt_train_wp, gt_val_wp = [], [], [], []
    polyp_train_np, polyp_val_np, gt_train_np, gt_val_np = [], [], [], []
    polyp_train, polyp_val, gt_train, gt_val = [], [], [], []
    polyp_train, gt_train, polyp_val, gt_val = [], [], [], []
    for i, file_ in enumerate(images_root): 
        polyp_class = int(file_.split('-')[0][-1])
        # polyp_class = 1
        # print(len(polyp_train_wp), len(polyp_train_np))
        if polyp_class == 1 and len(polyp_train_wp) < train_wp: 
            polyp_train_wp.append(file_)
            gt_train_wp.append(file_.replace("images", "masks"))
            # polyp_train.append(file_)
            # gt_train.append(file_.replace("images", "masks"))
        if polyp_class == 0 and len(polyp_train_np) < train_np: 
            polyp_train_np.append(file_)
            gt_train_np.append(file_.replace("images", "masks"))
            # polyp_train.append(file_)
            # gt_train.append(file_.replace("images", "masks"))
        if polyp_class == 1 and len(polyp_train_wp) >= train_wp  and len(polyp_val_wp) < val_wp: 
            polyp_val_wp.append(file_)
            gt_val_wp.append(file_.replace("images", "masks"))
            # polyp_val.append(file_)
            # gt_val.append(file_.replace("images", "masks"))
        if polyp_class == 0 and len(polyp_train_np) >= train_np  and len(polyp_val_np) < val_np: 
            polyp_val_np.append(file_)
            gt_val_np.append(file_.replace("images", "masks"))
            # polyp_val.append(file_)
            # gt_val.append(file_.replace("images", "masks"))

    polyp_train =  polyp_train_wp + polyp_train_np 
    gt_train    =  gt_train_wp + gt_train_np 
    polyp_val   =  polyp_val_wp + polyp_val_np 
    gt_val      =  gt_val_wp + gt_val_np 
    
    polyp_, poly_val, gt_rain, gt_vl = train_test_split(images_root, gts_root, test_size=0.25)
    print(" ", len(images_root),len(polyp_), len(poly_val), len(gt_rain), len(gt_vl))
    print("-",len(polyp_train) + len(polyp_val),  len(polyp_train), len(polyp_val), len(gt_train), len(gt_val))

    return polyp_train, polyp_val, gt_train, gt_val