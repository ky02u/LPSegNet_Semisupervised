import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import argparse
import torch 
import random 
import numpy as np
from lib.PraNet_Res2Net_v10 import AttentionNetwork, LPSegNet
from lib.PraNet_Res2Net_v7 import PraNet
from lib.HarDMSEG       import HarDMSEG
from utils.dataloader   import get_loader
from utils.data         import generate_train_test_split
from LPSegNet           import train

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type = int,   default = 20)
    parser.add_argument('--lr',          type = float, default = 1e-6)#1e-4 1e-6 finetunning True 1e-6 para pseudo
    parser.add_argument('--batchsize',   type = int,   default = 8)
    parser.add_argument('--trainsize',   type = int,   default = 352)
    parser.add_argument('--decay_rate',  type = float, default = 0.1)
    parser.add_argument('--decay_epoch', type = int,   default = 50)
    parser.add_argument('--train_path',  type = str,   default = '/data/jefelitman_pupils/danielortiz_tesis/danielortiz/lina/Polyps/tesismaestria/LPSegNet/dataset_2')
    parser.add_argument('--train_save',  type = str,   default = '80_public_semi')
    parser.add_argument('--pth_path',    type=str,     default='./snapshots_2/80_public_2/80.pth')
    parser.add_argument('--pretrained',  type=bool,    default=True)
    opt = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # build model
    # model       = LPSegNet().cuda()
    model       = PraNet(use_attention='PCM', mode_cls='max_pooling').cuda()
    # model       = HarDMSEG(use_attention='PCM', mode_cls='max_pooling').cuda()

    if opt.pretrained == True:
        print("Pretrained model")
        model.load_state_dict(torch.load(opt.pth_path))
        for param in model.parameters():
            param.requires_grad = True

    # params      = model.parameters()
    optimizer   = torch.optim.Adam(model.parameters(), opt.lr)
    image_root  = '{}/images/'.format(opt.train_path)
    gt_root     = '{}/masks/'.format(opt.train_path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using for training:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("Failed to find GPU, using CPU instead.")

    polyp_train, polyp_val, gt_train, gt_val =  generate_train_test_split(image_root, gt_root)

    train_loader = get_loader(polyp_train, gt_train, batchsize=opt.batchsize, trainsize=opt.trainsize)
    val_loader   = get_loader(polyp_val, gt_val, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step   = len(train_loader)
    model, loss  = train(train_loader, val_loader, model, False, optimizer, 
                        opt.epoch,  opt.trainsize, opt.lr, opt.decay_rate, opt.decay_epoch)

    # ----- save weights -----
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), save_path + '%d.pth' % opt.epoch)
    print('[Saving Snapshot:]', save_path + '%d.pth'% opt.epoch)