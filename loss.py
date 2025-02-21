import torch
import cv2
import numpy                as np
import matplotlib.pyplot    as plt
import torch.nn.functional  as F


def calculated_iou(pred, mask, images, label, name, fp): 

    pred = torch.sigmoid(pred)
    iou_, cont = 0, 0
    fp_, tn, fp_name = 0, 0, []
    for index, prediction in enumerate(pred):
        pred_ = torch.round(prediction)
        mask_ = torch.round(mask[index])
        label_ = label[index]
        name_ = name[index]
        image = images[index]
        sum_mask_ = torch.sum(mask_)
        predict_sum = torch.sum(pred_)
        N, W, H = pred_.size()       
        # tp = torch.sum(pred_*mask_)
        # tn = torch.sum((1-pred_)*(1-mask_))
        matrix = torch.add(pred_, mask_)
        inter_1s = torch.sum((matrix == 2.0))
        inter_0s = torch.sum((matrix == 0.0))
        union = torch.sum((matrix != 0.0))
        
        # if sum_mask_ == 0:
        #     iou = ((inter_0s)/ (W*H))
        # else: 
        iou = ((inter_1s)/ (union))

        if sum_mask_ == 0 and predict_sum == 0:
            iou = torch.tensor(1.0).cuda()
            tn += 1
        # if fp:
        #     save_fp(iou.item(), image, label_.item(), name_, pred_)

        if sum_mask_ == 0 and predict_sum != 0: 
            fp_ += 1
            fp_name.append(name)
        iou_ += iou
        cont += 1
    iou_t = iou_
    return iou_t, cont, tn, fp_
    
def structure_loss(pred, mask):
    dimensiones = (1,2)
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))
    
    pred = torch.sigmoid(pred)
    wiou_total = []

    for index, prediction in enumerate(pred):
        gt_sum = torch.sum(torch.round(mask[index]))
        predict_sum = torch.sum(torch.round(prediction))
        if gt_sum == 0 and predict_sum == 0:
            wiou = 0.0
        else:
            inter = ((prediction * mask[index])*weit[index]).sum(dim=dimensiones)
            union = ((prediction + mask[index])*weit[index]).sum(dim=dimensiones)
            wiou = 1.0 - (inter + 1)/(union - inter + 1)
            wiou = wiou.detach().cpu().numpy()[0]
        wiou_total.append(wiou)
    array = np.array(wiou_total)
    result = torch.tensor(array).cuda()
    return (wbce + result).mean()    # return result.mean()

def generate_weights_mask(gt):
    weight_per_class = torch.tensor([0.3, 0.7]).cuda()
    weights = torch.where(gt == 0.0, weight_per_class[0], weight_per_class[1])
    return weights

def balanced_loss(pred, mask):
    
    pred = torch.sigmoid(pred)

    loss = 0
    for index, prediction in enumerate(pred):
        gt_sum = torch.sum(mask[index])
        predict_sum = torch.sum(prediction)
        B, N, W, H = pred.size()
        total = W*H
        if gt_sum ==0 and predict_sum != 0: 
            w = ((predict_sum)/total)*10
        else: w = torch.tensor(0.0)
        
        # bce = F.binary_cross_entropy_with_logits(prediction, mask[index], reduce='none')
        l1 = torch.nn.L1Loss()
        metric = l1(prediction, mask[index])
        loss += (1+w)*metric

    loss_t = loss/len(pred)
    return  loss_t

def save_fp(iou, image, label, name, pred):
    new_img_path = './dataset_2/images/0-fp-' + name[:-4] + '.png'
    new_gt_path = './dataset_2/masks/0-fp-' + name[:-4] + '.png'

    c, w, h = image.shape
    image_np = image.detach().cpu().numpy()
    image_rgb = np.zeros((w,h,c))
    image_rgb[:,:,2] = image_np[0,:,:]*255
    image_rgb[:,:,1] = image_np[1,:,:]*255
    image_rgb[:,:,0] = image_np[2,:,:]*255

    
    pred = pred.detach().cpu().numpy()
    pred = pred[0]*255
    if (iou <= 0.5 and label == 1) or (label == 0 and iou < 1): 
        if  pred.mean() < 150: 
            cv2.imwrite(new_img_path, image_rgb)
            cv2.imwrite(new_gt_path, pred)