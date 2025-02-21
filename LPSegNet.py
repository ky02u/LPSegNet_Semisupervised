
import os
import torch
import torch.nn.functional  as F
from torch.autograd import Variable
from tqdm           import tqdm
from utils.utils    import clip_gradient, adjust_lr
from loss           import balanced_loss, calculated_iou, structure_loss
import matplotlib.pyplot as plt

def train(train_loader, val_loader, model, fp, optimizer, epoch, train_size, lr, decay_rate, decay_epoch):

    loss_train, loss_val, iou_train, iou_val = [], [], [], []
    f            = open ('logits.txt','a')
    size_rates   = [0.75, 1, 1.25]
    stream_train = tqdm(range(epoch))

    for epoch_ in stream_train:

        adjust_lr(optimizer, lr, epoch_, decay_rate, decay_epoch)

        loss_t, loss_v, iou_t, iou_v     = 0.0, 0.0, 0.0, 0.0
        total_t, total_v                 = 0.0, 0.0

        model.train()
        for i, sample in enumerate(train_loader, start=1):
            for rate in size_rates:

                # ---- data prepare ----
                images, gts, label, name = sample
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # label2 = label.squeeze(1)
                label2 = label.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                label2 = label2.type(torch.FloatTensor)
                label2 = Variable(label2.cuda())

                # ---- rescale ----
                trainsize = int(round(train_size*rate/32)*32)
                if rate != 1:
                    images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners = True)
                    gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners = True)

                # ----- forward step ---
                loss, total, iou = forward_step('train', model, images, optimizer, label2, name, gts, i, fp)
                loss_t += loss.detach().cpu().numpy()
                iou_t += iou.detach().cpu().numpy()
            total_t += total    

        model.eval()
        with torch.no_grad():
            for j, sample in enumerate(val_loader, 1):
                image, gts_v, label_, name_ = sample
                image = Variable(image).cuda()
                gts_v = Variable(gts_v).cuda()
                # label_v = label_.squeeze(1)
                label_v = label_.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                label_v = label_v.type(torch.FloatTensor)
                label_v = Variable(label_v.cuda())

                loss, total, iou = forward_step('validation', model, image, optimizer, label_v, name_, gts_v, j, fp)
                loss_v += loss.detach().cpu().numpy()
                iou_v += iou.detach().cpu().numpy()
                total_v += total

        iouTrain = round(100*iou_t/(total_t*3),2)
        iouVal   = round(100*iou_v/(total_v),2)

        stream_train.set_description("epoch: {} TrainLoss: {} \t ValLoss: {} \t IoUTrain: {} \t IoUVal: {} ".format(epoch_+1,
                                        round(loss_t/i,2), round(loss_v/j,2), iouTrain, iouVal))

        f.write("\nepoch: {}\t TrainLoss: {} \t ValLoss: {}  \t IoUTrain: {} \t IoUVal: {}".format(epoch_+1,
                    round(loss_t/i,2), round(loss_v/j,2), iouTrain, iouVal))

        loss_train.append(loss_t/i)
        loss_val.append(loss_v/j)
        iou_train.append(iouTrain)
        iou_val.append(iouVal)

        # ----- save weights -----
        # save_path = 'snapshots/KQV_29/'
        # os.makedirs(save_path, exist_ok=True)
        # torch.save(model.state_dict(), save_path + '%d.pth' % epoch_)
        # print('[Saving Snapshot:]', save_path + '%d.pth'% epoch_)

    print('Training Complete!')
    f.close()
    return model, loss_train

def forward_step(mode, model, images, optimizer, label2, name,  gts, i, fp):
    
    # ---- forward ----
    loss = 0
    optimizer.zero_grad()
    pcm, classifcation = model(images)

    loss_cls           = F.binary_cross_entropy_with_logits(classifcation, label2, reduce='none')
    loss_contorno      = structure_loss(pcm, gts)
    iou, cont, tn, fp_ = calculated_iou(pcm, gts, images, label2, name, fp)    

    # ---- backward ----
    total = images.size(0)
    loss  = loss_contorno + (1-(iou)/total) + loss_cls 

    if mode == 'train':
        loss.backward()
        clip_gradient(optimizer, grad_clip = 0.5)
        optimizer.step()
    # else: 
        # plt.imsave("val/rfb.png",  torch.round(torch.sigmoid(pcm)).detach().cpu().numpy()[0][0], cmap = 'gray')
    return loss, total, iou