import numpy as np
import torch

def metric_IOU(pred, mask, label):
    if label == 0:
      w, h = mask.shape      
      inter = np.sum((w*h)) - np.sum((pred + mask))
      union = np.sum((w*h))
      iou = inter/union

    else:
        inter = np.sum((pred * mask))
        union = np.sum((pred + mask))
        iou = inter/(union-inter)
    iou = iou*100
    iou = round(iou,2)
    return  iou

def calculated_iou(pred, mask): 

    pred = torch.sigmoid(pred)

    pred_ = torch.round(pred)
    mask_ = torch.round(mask)
    
    matrix_iou = torch.cat((pred_, mask_), dim=0)
    matrix = torch.sum(matrix_iou, dim = 0)


    if torch.sum(pred_) == 0 and torch.sum(mask_) == 0:
        iou = torch.tensor(1.0)
    else: 
        inter = torch.sum((matrix == 2.0))
        union = torch.sum((matrix != 0.0))
        iou = inter /union 
    return iou

def metrics_cls(preds, label):
    TP, FP, TN, FN = [], [], [], []
    tp, fp, tn, fn = 0, 0, 0, 0
    for index, pred in enumerate(preds):
        if pred == 1 and label[index] == 1:  
            tp += 1
            TP.append(1)
        if pred == 1 and label[index] == 0:  
            fp += 1
            FP.append(1)
        if pred == 0 and label[index] == 1:   
            fn += 1
            FN.append(1)
        if pred == 0 and label[index] == 0: 
            tn += 1
            TN.append(1)

    print("\nCLASSIFICATION")
    accuracy, precision, recall = calculate_metrics(TP, FP, TN, FN)

    return  accuracy, precision, recall

def metrics_seg(preds, label, dataset): 
  tp, fp, tn, fn = 0, 0, 0, 0
  TP, FP, TN, FN = [], [], [], []
  for index, pred in enumerate(preds):
    if pred >= 50 and label[index] == 1:  
        tp = 1
        TP.append(tp)
    if pred < 100 and label[index] == 0:  
        fp = 1
        FP.append(fp)
    if pred < 50 and label[index] == 1:   
        fn = 1
        FN.append(fn)
    if pred == 100 and label[index] == 0: 
        tn = 1
        TN.append(tn)
  
  print("\nSEGMENTATION " + dataset + "\n" )
  accuracy, precision, recall, spec = calculate_metrics(TP, FP, TN, FN)
  
  return  accuracy, precision, recall, spec

def calculate_metrics(TP, FP, TN, FN):
  TP, FP, TN, FN = np.array(TP), np.array(FP), np.array(TN), np.array(FN)
  FP, TP, FN, TN, total = FP.sum(), TP.sum(), FN.sum(), TN.sum(), FP.sum()+ TP.sum()+ FN.sum()+TN.sum()
  accuracy, precision, recall, spec = (TP+TN)/(TP+TN+FP+FN), TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)
  accuracy, precision, recall, spec = round(accuracy*100,2), round(precision*100,2), round(recall*100,2), round(spec*100,2)
  print("FP:\t", FP, "\nTP:\t", TP, "\nFN:\t", FN, "\nTN:\t", TN)
  print("\nAccuracy:\t", accuracy, "\nPrecision:\t", precision, "\nRecall:\t\t", recall, "\nSpec:\t\t", spec)

  return accuracy, precision, recall, spec

def plot_confussion_matrix(y_test, y_pred, name):
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import os 

    save_path = 'results/' + name
    os.makedirs(save_path, exist_ok=True)
    
    cf_matrix = confusion_matrix(y_test, y_pred)
    
    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ');

    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    fig = ax.get_figure()
    fig.savefig(save_path + '/confusion_matrix.png')