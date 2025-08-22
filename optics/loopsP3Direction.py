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
    labels_idx = torch.argmax(labels, dim=1).float()
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
               IP,
               save_dict, device, 
               optimizer = None, 
               scheduler = None, 
               train =True):	# Train and Val loops. Default is train
    
    model = model #.
    total_samples = len(data)
    """if optimizer: # need a choose scheduler function!
        print("Optimizer present: ",optimizer)
        scheduler = choose_scheduler(save_dict, optimizer)"""
    if train:
        model.train()
    else:
        model.eval()   
    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0
    labels =[]
    batch_acc_MSE = []
    batch_acc_MAE = []
    batch_peakdist = []
    img_batch = None
    imNorm_batch = None
    numBatch = 0
    sizeBatch = 0
    #print("loopBatch pre loop- Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
    
    for i, batch in enumerate(data,0):
        #print(f"len of batch : {len(batch)}") # we arte getting a train loss for each batch 
        
        #print(f"{i}  S batch")
        #print(f"len data  {len(data)}")
        x_batch, y_batch, img_batch, imNorm_batch = batch #, img_batch, imNorm_batch

        numBatch = len(data)
        sizeBatch += len(x_batch)
        
        if sizeBatch ==0 or numBatch == 0:
            print(f"{i} sizeBatch: {sizeBatch}   numBatch:  {numBatch}")
        #print(f"{i} expected num batcheds: {numBatch}")
        #print(f"{i} len of x_batch  {len(x_batch)}")
        
        prediction = model.forward(x_batch.to(device))
        #print("prediction made - Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        loss = loss_fn(prediction, y_batch.to(device))
        #print("loss calculated- Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        #print('prediction:    ', prediction.shape)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler and scheduler is not "NoSched":
                scheduler.step(loss)
            
        for j in range(len(y_batch)-1):
            if y_batch[j].argmax() == prediction[j].argmax():
                num_correct +=1
        # img, scale:int, loop_run_name:str, save_dict:dict,  epoch:int, where:str

        
        """randomval = random.randint(0, len(x_batch))
        if i == randomval:
            print('in loop')
            IP.view(x_batch[0], 1, None, None, None, None)
            print(x_batch[0])"""

        
        [predict_list.append(pred.argmax().to('cpu').item()) for pred in prediction]# .argmax()
        [labels.append(y.argmax().to('cpu').item()) for y in y_batch] #.argmax()

        total_count+= batch_size
        current_loss += loss.item()

        #print("loopBatch end of loop Current allocated memory (GB):", torch.cuda.memory_allocated(device=device) / 1024 ** 3)
        
            
        acc_MSE = MSE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_acc_MSE.append(acc_MSE)
        acc_MAE =  MAE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_acc_MAE.append(acc_MAE)
        peakdist = peak_disterr_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_peakdist.append(peakdist)

        #print('accuracy MSE: ', acc_MSE )
        #print('accuracy MAE: ', acc_MAE)
        #print(f"accuracy peak dist  err {peakdist}")

        
        wandb.log({'train_acc_MSE':acc_MSE})
        wandb.log({'train_acc_MAE':acc_MAE})
        wandb.log({'train_peakDistErr': peakdist})
        #print(f"{i} E batch")

    #print(f" E looop")
    if sizeBatch ==0 or numBatch == 0:
        print(f" sizeBatch: {sizeBatch}   numBatch:  {numBatch}")
    sizeBatch = sizeBatch / numBatch # get the average batch size

    if len(batch_acc_MSE) != numBatch:
        print(f"You're maths logic was faulty!", numBatch, len(batch_acc_MSE))
        print(batch_acc_MSE)

    batch_acc_MSE_mean = (sum(batch_acc_MSE) / len(batch_acc_MSE))
    batch_acc_MAE_mean = (sum(batch_acc_MAE) / len(batch_acc_MAE))
    batch_peakdist_mean = (sum(batch_peakdist) / len(batch_peakdist))
    acc = {'MSE':batch_acc_MSE_mean, 'MAE':batch_acc_MAE_mean, 'peakDist':batch_peakdist_mean}
    if train:
        return current_loss, predict_list, labels, num_correct, acc, model, optimizer, img_batch, imNorm_batch #, lr_ls
    else:
        return current_loss, predict_list, labels, num_correct, acc, img_batch, imNorm_batch # changed y_batch to labels in return 



def test_loop_batch(model,data, loss_fn, batch_size, device):
    import sys
    from plottingP3Direction import plot_predictions
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
            tense, label, img_batch, imNorm_batch = batch #, img_batch, imNorm_batch
            #print("in test batch. got tense and label from batch. type len.  tese:", type(tense), len(tense), " label:", type(label), len(label))
            #print(label)

            
            prediction = model.forward(tense.to(device))
            for i in range(len(label)-1):
                #print(len(label), label[0].argmax(), len(label)-1)
                if label[i].argmax() == prediction[i].argmax():
                    num_correct +=1
            [predict_list.append(pred.to('cpu')) for pred in prediction]  #.argmax()  # .argmax(),.item(),.argmax(),.item()
            [label_list.append(lab.to('cpu')) for lab in label] #.argmax()  # .argmax(),.item(),.argmax(),.item()
            #print("in test bAtch post list comprehension. pred:", len(predict_list), "lab:", len(label_list))
            total_count += batch_size
            #correct +=(prediction.argmax()==label.argmax()).sum().item()
        #acc = num_correct/total_count
        #accuracy = 100*(acc)
        plot_predictions(prediction, label, num_samples=len(tense))
        
        test_acc_MSE = MSE_metric(prediction.to('cpu'), label.to('cpu'))
        test_acc_MAE =  MAE_metric(prediction.to('cpu'), label.to('cpu'))
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
    #model.train()
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

    if optimizer: # need a choose scheduler function!
        print("Optimizer present: ",optimizer)
        scheduler = choose_scheduler(save_dict, optimizer)
        
    for epoch in tqdm(range(save_dict['start_epoch'],epochs)):

        random_value = random.randrange(0,batch_size)
        print('Training...')
        # , img_batch, imNorm_batch

        t_loss, train_prediction, t_label_list, t_correct, tacc, model, optimizer, img_batch, imNorm_batch = loop_batch(model, 
                                                                                                                  train,
                                                                                                                  loss_fn,
                                                                                                                  batch_size,
                                                                                                                  sample, 
                                                                                                                  random_value, 
                                                                                                                  epoch, 
                                                                                                                  loop_run_name, 
                                                                                                                  IP,
                                                                                                                  save_dict = save_dict, 
                                                                                                                  device = device, 
                                                                                                                  optimizer = optimizer, 
                                                                                                                  scheduler = scheduler_value, 
                                                                                                                  train = True) 
        
        #imNormBatch_list.append(imNorm_batch)

        print("tacc: ",tacc)
        if int(epoch) == int(random_value):     # == 0 and epoch >1:
            print(f"EPOCH    {epoch} / {epochs}:")
            IP.view2(img_batch[0], 1, "original")
            IP.view2(imNorm_batch[0], 1, "Processed") # img, scale:int, name:str
            print(f"TRUE LABEL    :          {t_label_list[0]}")
            print(f"PREDICTION    :          {train_prediction[0]}")

        #IP.view2(img_batch[6], 1, "original")
        #IP.view2(imNorm_batch[6], 1, "Processed") # img, scale:int, name:str
        #print(f"TRUE LABEL    :          {t_label_list[6]}")
        #print(f"PREDICTION    :          {train_prediction[6]}")
        
        t_loss_list.append(t_loss)
        #[t_predict_list.append(pred.argmax()) for pred in train_prediction]
        #print(f"prediction    {train_prediction[0]}, {type(train_prediction[0])}")
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
        #print(epoch,len(val))
        #!nvidia-smi
        # , img_batch, imNorm_batch
        v_loss, val_prediction, v_label_list, val_correct, vacc, img_batch, imNorm_batch = loop_batch(model, 
                                                                                                val, 
                                                                                                loss_fn,
                                                                                                batch_size,
                                                                                                sample,
                                                                                                random_value,
                                                                                                epoch,
                                                                                                loop_run_name, 
                                                                                                IP,
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

