import os
import cv2
import csv

import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import argparse
import torch
import torch.nn             as nn
import numpy                as np
import matplotlib.pyplot    as plt
from lib.PraNet_Res2Net_v10 import AttentionNetwork, LPSegNet
from lib.PraNet_Res2Net_v7  import PraNet
from utils.dataloader       import test_dataset
from metrics                import metric_IOU, metrics_seg
from activations            import show_activations, generate_segmentation, generate_segmentation_numpy


def guardar_mascara(output, nombre_archivo, carpeta_destino):
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)
    ruta_destino = os.path.join(carpeta_destino, nombre_archivo)
    np.save(ruta_destino, output)
    print(f"Máscara guardada en: {ruta_destino}")

def guardar_iou_csv(nombres_archivos, iou, nombre_archivo_csv):
    with open(nombre_archivo_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Nombre de archivo', 'IoU'])
        for nombre, iou_valor in zip(nombres_archivos, iou):
            writer.writerow([nombre, iou_valor])
    print(f"Datos de IoU guardados en: {nombre_archivo_csv}")



def extract_name(name):
    if len(name) == 5:
        new_name = '000' + name
    elif len(name) == 6: 
        new_name = '00' + name
    elif len(name) == 7: 
        new_name = '0' + name
    else: 
        new_name = name
    
    return new_name

def extract_name_number(name):
    if name <= 9:
        new_name = '00' + str(name)
    elif name>9 and name<=99: 
        new_name = '0' + str(name)
    else: 
        new_name = str(name)
    return new_name

def save_outputs(outputs, output_folder):    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, output in enumerate(outputs):
        output_path = os.path.join(output_folder, f"output_{i}.png")
        output = (output * 255).astype(np.uint8)
        cv2.imwrite(output_path, output)


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data/jefelitman_pupils/danielortiz_tesis/danielortiz/polyps_semi/LPSegNet/snapshots_2/IGHO_FederadoLocal_Ronda20/20.pth')
parser.add_argument('--dataset', type=str, default='images')
opt = parser.parse_args()
#parser.add_argument('--pth_path', type=str, default='/data/jefelitman_pupils/danielortiz_tesis/danielortiz/polyps_semi/LPSegNet/snapshots_2/federado_ponderado/Federado_Ponderado_Ronda19.pth')
#/data/jefelitman_pupils/danielortiz_tesis/danielortiz/polyps_semi/LPSegNet/snapshots_2/60_public_prom_4.0/20.pth
#='/data/jefelitman_pupils/Promedio_Federado.pth'
# Datasets y paths
datasets =  {
    'images': ['Kvasir', 'ETIS-Larib', 'CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'Kvasir'],
    'IGHO-AGO': ['2021-08-26_105781', '2021-08-30_106015_2', '2021-08-30_106025'],
    'IGHO-NOV': ['2021-11-25_134924_312', '2021-11-25_143810_249', '2021-11-25_151631_034', '2021-11-25_175942_606'],
    'IGHO-SEPT': ['2021-09-02-106236', '2021-09-02-106239', '2021-09-02-106262'],
    'IGHO-JUN': ['2022-06-09_112525_362'],
    'CVC-video': ['Video1', 'Video2', 'Video3', 'Video4', 'Video5', 'Video6', 'Video7', 'Video8', 'Video9', 'Video10', 'Video11', 'Video12', 'Video13', 'Video15', 'Video16', 'Video17', 'Video18'],
    'ASU-Mayo': ['Video_2', 'Video_4', 'Video_24', 'Video_49', 'Video_52', 'Video_61', 'Video_66', 'Video_68', 'Video_69', 'Video_70'],
    'test': ['Test-CVC']
}

path = {
    'images': '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/data/TestDataset/',
    'IGHO-AGO': '/data/jefelitman_pupils/Igho/polipo_2021_AGOSTO/',
    'IGHO-SEPT': '/data/jefelitman_pupils/Igho/polipo_2021_SEPTIEMBRE/',
    'CVC-video': '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/data/TestDataset/CVC-Video/',
    'ASU-Mayo': '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/data/TestDataset/ASU-Mayo/',
    'IGHO-NOV': '/data/jefelitman_pupils/Igho/polipo_2021_NOVIEMBRE/',
    'IGHO-JUN': '/data/jefelitman_pupils/Igho/polipo/',
    'test': '/data/jefelitman_pupils/'
}

hist_path = '/home/linamruiz/Polyps/data/TestDataset/CVC_cls/valid.csv'
hist = False
model_name = opt.pth_path.split('/')[7]

labels_total = []
iou_seg_total, iou_total = [], []

# Para guardar los datos de IoU
iou_data = []


for index, _data_name in enumerate(datasets[opt.dataset]):
    iou_video, label_video, iou_seg_video = [], [], []
    print(_data_name)
    data_path = path[opt.dataset] + '{}'.format(_data_name)
    save_path = './results_CVC_video/' + model_name + '/{}/'.format(_data_name)
    
    os.makedirs(save_path, exist_ok=True)

    model = PraNet(use_attention='PCM', mode_cls='max_pooling').cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    
    model.cuda()
    model.eval()

    #image_root  = '{}/images/'.format(data_path)      
    #gt_root     = '{}/masks/'.format(data_path)       

    image_root  = '{}/Normal/'.format(data_path)      
    gt_root     = '{}/MASKS/'.format(data_path)           

    test_loader = test_dataset(image_root, gt_root, opt.testsize, hist, hist_path)
    count = 1

    for i in range(test_loader.size):
        image, gt, name_img, label, label_histhology = test_loader.load_data()
        
        gt = np.asarray(gt, np.float32)
        image2 = np.asarray(image, np.float32)
        label = 1 if gt.max() == 255.0 else 0
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        seg, classification = model(image)
        output, res_sigmoid = generate_segmentation_numpy(seg, gt, False)

        # Calcular IoU individual
        iou_individual = round(metric_IOU(output, gt, label), 2)
        iou_data.append((opt.dataset, _data_name, name_img, iou_individual, label))        

        iou_seg_total.append(iou_individual)
        iou_seg_video.append(iou_individual)
        labels_total.append(label)
        label_video.append(label)
        if label == 1: 
            iou_total.append(iou_individual)
            iou_video.append(iou_individual)
        count += 1

    iou_video = np.array(iou_video)
    iou_seg_video = np.array(iou_seg_video)
    label_video = np.array(label_video)
    acc, prec, recall, spec = metrics_seg(iou_seg_video, label_video, _data_name)
    print("IoU:\t\t", round(iou_video.mean(), 2))
    np.save(save_path + 'iou_wp.npy', iou_video)
    np.save(save_path + 'iou_np_wp.npy', iou_seg_video)
    np.save(save_path + 'iou_label.npy', label_video)

csv_save_path = '/home/jefelitman_pupils/data/jefelitman_pupils/danielortiz_tesis/danielortiz/polyps_semi/LPSegNet/CSV_AUC'
csv_file_name = 'Nov_public_60_IGHO.csv'
full_csv_path = os.path.join(csv_save_path, csv_file_name)

with open(full_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Conjunto de datos', 'Subconjunto', 'Nombre de archivo', 'IoU', 'Label'])
    for dataset_name, subset_name, nombre_img, iou_individual, label in iou_data:
        writer.writerow([dataset_name, subset_name, nombre_img, iou_individual, label])

iou_total = np.array(iou_total)
iou_seg_total = np.array(iou_seg_total)
labels_total = np.array(labels_total)
acc, prec, recall, spec = metrics_seg(iou_seg_total, labels_total, _data_name)
print("IoU T:\t\t", round(iou_total.mean(), 2))