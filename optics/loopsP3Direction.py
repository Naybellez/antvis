from functions import ImageProcessor
import torch

import cv2
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as maths

from scipy.signal.windows import gaussian


import os
import random

from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from torch.nn import functional

from tqdm import tqdm

import wandb

import sys
sys.path.append('../.')


from modelManagment import choose_scheduler


# assessment functions

# MSE # same as loss but this is to be held on to for human eyes
def MSE_metric(preds, labels):
    return torch.mean((preds-labels)**2).item()

# MAE # similar to above but absolute error. may provide wider understanding
def MAE_metric(preds, labels):
    return torch.mean(torch.abs(preds-labels)).item()
    
# peak distance error. # distance between the two gaus peaks (one for true labels and one for predictions)
def peak_disterr_metric(preds, labels):
    pred_idx = torch.argmax(preds, dim=1).float()
    labels_idx =torch.argmax(labels, dim=1).float()
    return torch.mean(torch.abs(pred_idx-labels_idx)).item()

# sub pixel  peak precision  #  quadratic interpolation around the maximum to estimate the true peak position
#def peakpos_metric(pred)

def loop_batch(model, 
               data, 
               loss_fn, 
               batch_size, 
               sample,
               random_value,
               epoch,
               loop_run_name, 
               save_dict, device, 
               optimizer = None, 
               scheduler = None, 
               train =True):	# Train and Val loops. Default is train
    
    model = model #.
    total_samples = len(data)
    if optimizer: # need a choose scheduler function!
        print("Optimizer present: ",optimizer)
        scheduler = choose_scheduler(save_dict, optimizer)
    if train:
        model.train()
    else:
        model.eval()   
    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0
    labels =[]
    #print("loopBatch pre loop- Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)

    for i, batch in enumerate(data,0):

        x_batch, y_batch, img_batch, imNorm_batch = batch
        
        prediction = model.forward(x_batch.to(device))
        #print("prediction made - Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        loss = loss_fn(prediction, y_batch.to(device))
        #print("loss calculated- Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        for i in range(len(y_batch)-1):
            if y_batch[i].argmax() == prediction[i].argmax():
                num_correct +=1
                

        [predict_list.append(pred.argmax().item()) for pred in prediction]
        [labels.append(y.argmax().item()) for y in y_batch]

        total_count+= batch_size
        current_loss += loss.item()

        #print("loopBatch end of loop Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        if scheduler and scheduler is not "NoSched":
            scheduler.step(loss)
            
        acc_MSE = MSE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        acc_MAE =   MAE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        peakdist = peak_disterr_metric(prediction.to('cpu'), y_batch.to('cpu'))

        #print('accuracy MSE: ', acc_MSE )
        #print('accuracy MAE: ', acc_MAE)
        #print(f"accuracy peak dist  err {peakdist}")

        acc = {'MSE':acc_MSE, 'MAE':acc_MAE, 'peakDist':peakdist}
        wandb.log({'train_acc_MSE':acc_MSE})
        wandb.log({'train_acc_MAE':acc_MAE})
        wandb.log({'train_peakDistErr': peakdist})

    if train:
        return current_loss, predict_list, labels, num_correct, acc, model, optimizer, img_batch, imNorm_batch #, lr_ls
    else:
        return current_loss, predict_list, labels, num_correct, acc, img_batch, imNorm_batch # changed y_batch to labels in return 


def plot_predictions(preds, targets, num_samples=5):

    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    plt.figure(figsize=(10, num_samples * 2))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(targets[i], label="Target", color='black', linewidth=2)
        plt.plot(preds[i], label="Prediction", color='red', linestyle='--')
        plt.title(f"Sample {i} | Target Peak : {torch.argmax(targets[i]).item()} | Pred Peak : {torch.argmax(preds[i]).item()}")
        plt.legend()
    plt.tight_layout()
    plt.show()

def test_loop_batch(model,data, loss_fn, batch_size, device):
    import sys
    sys.path.append('../.')
    model = model.eval()
    predict_list = []
    label_list = []
    total_count =0
    num_correct = 0
    correct = 0

    with torch.no_grad():
        for i, batch in enumerate(data,0):
            #tense = tense.to(device)
            tense, label, img_batch, imNorm_batch = batch
            #print("in test batch. got tense and label from batch. type len.  tese:", type(tense), len(tense), " label:", type(label), len(label))
            #print(label)

            
            prediction = model.forward(tense.to(device))
            for i in range(len(label)-1):
                #print(len(label), label[0].argmax(), len(label)-1)
                if label[i].argmax() == prediction[i].argmax():
                    num_correct +=1
            [predict_list.append(pred.argmax().to('cpu').item()) for pred in prediction]
            [label_list.append(lab.argmax().to('cpu').item()) for lab in label]
            #print("in test bAtch post list comprehension. pred:", len(predict_list), "lab:", len(label_list))
            total_count += batch_size
            #correct +=(prediction.argmax()==label.argmax()).sum().item()
        #acc = num_correct/total_count
        #accuracy = 100*(acc)
        plot_predictions(prediction, label)
        
        test_acc_MSE = MSE_metric(prediction.to('cpu'), label.to('cpu'))
        test_acc_MAE =   MAE_metric(prediction.to('cpu'), label.to('cpu'))
        test_peakdist = peak_disterr_metric(prediction.to('cpu'), label.to('cpu'))

        #print('test accuracy MSE: ', test_acc_MSE )
        #print('test accuracy MAE: ', test_acc_MAE)
        #print(f"test accuracy peak dist  err {test_peakdist}")
        
        wandb.log({'test_acc_MSE':test_acc_MSE})
        wandb.log({'test_acc_MAE':test_acc_MAE})
        wandb.log({'test_peakDistErr': test_peakdist})
        
        #print(accuracy)
        #wandb.log({'test_acc': accuracy})
        accuracy = {'MSE': test_acc_MSE, 'MAE': test_acc_MAE, 'peakDist': test_peakdist}
        return accuracy, predict_list, label_list

def train_val_batch(model, train, val, loop_run_name, save_dict, lr, loss_fn, epochs, batch_size, optimizer, scheduler_value, device): #train_dl, val_dl, 
    #print("Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 ** 3) 
    import sys
    sys.path.append('../.')
    import pickle
    #import wandb
    from IPython.display import clear_output
    IP = ImageProcessor(device)
    model.train()
    t_loss_list = []
    v_loss_list = []
    t_predict_list = []
    v_predict_list = []
    t_accuracy_list = []
    v_accuracy_list = []
    t_label_list = []
    v_label_list = []
    imNormBatch_list = []
    #labels = []
    sample = False
    
    total_epochs = 0
    #print("Before Epochs of training - Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
    for epoch in tqdm(range(save_dict['start_epoch'],epochs)):

        random_value = random.randrange(0,batch_size)
        print('Training...')

        t_loss, train_prediction, t_label_list, t_correct, tacc, model, optimizer, img_batch, imNorm_batch = loop_batch(model, 
                                                                                                                  train,
                                                                                                                  loss_fn,
                                                                                                                  batch_size,
                                                                                                                  sample, 
                                                                                                                  random_value, 
                                                                                                                  epoch, 
                                                                                                                  loop_run_name, 
                                                                                                                  save_dict, 
                                                                                                                  device, 
                                                                                                                  optimizer = optimizer, 
                                                                                                                  scheduler = scheduler_value, 
                                                                                                                  train = True) 
        
        #imNormBatch_list.append(imNorm_batch)
        print(f"EPOCH    {epoch} / {epochs}:")
        IP.view2(img_batch[0], 1, "original")
        IP.view2(imNorm_batch[0], 1, "Processed") # img, scale:int, name:str
        print(f"TRUE LABEL    :          {t_label_list[0]}")
        print(f"PREDICTION    :          {train_prediction[0]}")

        IP.view2(img_batch[6], 1, "original")
        IP.view2(imNorm_batch[6], 1, "Processed") # img, scale:int, name:str
        print(f"TRUE LABEL    :          {t_label_list[6]}")
        print(f"PREDICTION    :          {train_prediction[6]}")
        
        t_loss_list.append(t_loss)
        #[t_predict_list.append(pred.argmax()) for pred in train_prediction]
        t_predict_list.append(train_prediction)
        wandb.log({'t_loss':t_loss})
        t_accuracy_list.append(tacc)

        #train_acc = (t_correct/(len(train)*batch_size)*100) ###
        # MSE_metric    MSE_metric   peak_disterr_metric
        #print(f"LAB  {type(t_label_list)},   {t_label_list}")
        #print(f" PRED   {type(train_prediction)},   {train_prediction}")
        #train_acc_MSE = MSE_metric(train_prediction, t_label_list)
        #train_acc_MAE =   MAE_metric(train_prediction, t_label_list)
        #train_peakdist = peak_disterr_metric(train_prediction, t_label_list)

        #print('train accuracy MSE: ', train_acc_MSE )
        #print('train accuracy MAE: ', train_acc_MAE)
        #print(f"train accuracy peak dist  err {train_peakdist}")

        #train_acc = {'MSE':train_acc_MSE, 'MAE':train_acc_MAE, 'peakDist':train_peakdist}
        #t_accuracy_list.append(train_acc)
        #wandb.log({'train_acc_MSE':train_acc_MSE})
        #wandb.log({'train_acc_MAE':train_acc_MAE})
        #wandb.log({'train_peakDistErr': train_peakdist})
        

        print('Validating...')
        #!nvidia-smi
        
        v_loss, val_prediction, v_label_list, val_correct, vacc, img_batch, imNorm_batch = loop_batch(model, 
                                                                                                val, 
                                                                                                loss_fn,
                                                                                                batch_size,
                                                                                                sample,
                                                                                                random_value,
                                                                                                epoch,
                                                                                                loop_run_name, 
                                                                                                save_dict, 
                                                                                                device, 
                                                                                                optimizer = None, 
                                                                                                scheduler = None, 
                                                                                                train = False)
        v_loss_list.append(v_loss)
        #[v_predict_list.append(pred) for pred in val_prediction]
        v_predict_list.append(val_prediction)
        wandb.log({'v_loss':v_loss})
        
        #val_acc = (val_correct/(len(val)*batch_size)*100)
        #val_acc_MSE = MSE_metric(val_prediction, v_label_list)
        #val_acc_MAE =   MAE_metric(val_prediction, v_label_list)
        #val_peakdist = peak_disterr_metric(val_prediction, v_label_list)

        #print('val accuracy MSE: ', val_acc_MSE )
        #print('val accuracy MAE: ', val_acc_MAE)
        #print(f"Val accuracy peak dist  err {val_peakdist}")

        #val_acc = {'MSE':val_acc_MSE, 'MAE':val_acc_MAE, 'peakDist': val_peakdist}
        v_accuracy_list.append(vacc)

        #wandb.log({'val_acc_MSE':val_acc_MSE})
        #wandb.log({'val_acc_MAE':val_acc_MAE})
        #wandb.log({'val_peakDistErr': val_peakdist})


        #v_accuracy_list.append(val_acc)
        #print('validation accuracy: ', val_acc )
        #wandb.log({'val_acc':val_acc})
        wandb.log({'c_epoch':epoch})
        total_epochs += 1
        #print(f"After Epoch {total_epochs} - Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 ** 3)
        #if epoch %50==0 and epoch !=0 and epoch != int(save_dict['start_epoch']):

        #clear_output()
        
    save_dict['Current_Epoch'] = epochs
    save_dict['training_samples'] = len(train)
    save_dict['validation_samples'] = len(val)
    
    save_dict['t_accuracy_list'] = t_accuracy_list 
    save_dict['v_accuracy_list'] = v_accuracy_list  #

    save_dict['t_loss_list'] = t_loss_list
    save_dict['v_loss_list'] = v_loss_list
    
    save_dict['t_labels'] = t_label_list
    save_dict['v_labels'] = v_label_list
    
    save_dict['t_predict_list'] = t_predict_list 
    save_dict['v_predict_list'] = v_predict_list  #
    
    return model, save_dict

