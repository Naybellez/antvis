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
#import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from torch.nn import functional


from tqdm import tqdm

import wandb

import sys
sys.path.append('../.')

from torch.utils.data import DataLoader
from dataloaderP3Direction import IDSWDataSetLoader3




# assessment functions

# MSE # same as loss but this is to be held on to for human eyes
def MSE_metric(preds, labels):
    return torch.mean((preds-labels)**2).item()

# MAE # similar to above but absolute error. may provide wider understanding
def MAE_metric(preds, labels):
    return torch.mean(torch.abs(preds-labels)).item()

def MAE_metric2(preds, labels, num_classes = 360):
    return torch.minimum(torch.mean(torch.abs(preds-labels)).item(), num_classes - torch.mean(torch.abs(preds-labels)).item())
    
# peak distance error. # distance between the two gaus peaks (one for true labels and one for predictions)
def peak_disterr_metric1(preds, labels):
    pred_idx = torch.argmax(preds, dim=1).float()
    labels_idx = torch.argmax(labels, dim=1).float()
    return torch.mean(torch.abs(pred_idx-labels_idx)).item()

def peak_disterr_metric2(preds, labels):
    #print(f"peakdist2. {preds.shape}") # is batch of 32     torch.Size([32, 360])
    #print("Labels shape : ",labels.shape) # is batch of 32  torch.Size([32, 360])
    #print("Labels : ",labels)
    #print("predictions", preds)
    
    pred_idx = torch.argmax(preds, dim=1).float()
    labels_idx = torch.argmax(labels, dim=1).float()
    num_classes = labels.shape[1]
    
    # Absolute difference
    diff = torch.abs(pred_idx - labels_idx)
    # Wrap around circle
    wrapped_diff = torch.minimum(diff, num_classes - diff) # to account for circular labels. return the smaller val (diff OR num_classes- diff)
    
    return [w.item() for w in wrapped_diff], torch.mean(wrapped_diff).item()

"""
for j in range(len(y_batch)-1):
            if y_batch[j].argmax() == prediction[j].argmax():
                num_correct +=1

"""




# get pred.argmax() 
# is the index within the gauss of label
def get_roughAcc(plusMinus, Tlabel, Preds):
    num_tries = 0
    num_correct = 0
    Tlabel = Tlabel.to('cpu')
    Preds = Preds.to('cpu')
    for j in range(len(Tlabel)): # -1
        labelrange = np.argwhere(Tlabel[j] > 0)
        if Preds[j].argmax() in labelrange:
            num_correct += 1
        num_tries += 1
    acc = (num_tries, num_correct, (num_correct/num_tries)*100)
    return acc

# sub pixel  peak precision  #  quadratic interpolation around the maximum to estimate the true peak position
#def peakpos_metric(pred)

def loop_batch(model, 
               data, 
               loss_fn, 
               batch_size, 
               sample,
               random_value,
               epoch, 
               IP,
               save_dict, device, config,
               optimizer = None, 
               train =True):	# Train and Val loops. Default is train
    
    model = model #.
    total_samples = len(data)
    if optimizer: # need a choose scheduler function!
        print("Optimizer present: ",optimizer)
        #scheduler = choose_scheduler(save_dict, optimizer)#"NoSched"#"RoP"#"Exp"
    
    if train:
        model.train()
    else:
        model.eval()   
    predict_list = []
    total_count = 0
    #num_correct = 0
    
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
        x_batch, y_batch, img_batch, imNorm_batch = batch #, img_batch, imNorm_batch

        numBatch = len(data)
        sizeBatch += len(x_batch)
        
        if sizeBatch ==0 or numBatch == 0:
            print(f"{i} sizeBatch: {sizeBatch}   numBatch:  {numBatch}")

        
        prediction = model.forward(x_batch.to(device))
        loss = loss_fn(prediction, y_batch.to(device))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
          
        acc = get_roughAcc(config.std_dev, y_batch, prediction)
        if train:
            wandb.log({"TrainAcc rough": acc[2]})
        else:
            wandb.log({"ValAcc rough": acc[2]})

        
        """randomval = random.randint(0, len(x_batch))
        if i == randomval:
            print('in loop')
            IP.view(x_batch[0], 1, None, None, None, None)
            print(x_batch[0])"""
    
        
        [predict_list.append(pred.argmax().item()) for pred in prediction]# .argmax()  .to('cpu')
        [labels.append(y.argmax().item()) for y in y_batch] #.argmax()  .to('cpu')

        total_count+= batch_size
        current_loss += loss.item()

        acc_MSE = MSE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_acc_MSE.append(acc_MSE)
        acc_MAE =  MAE_metric(prediction.to('cpu'), y_batch.to('cpu'))
        batch_acc_MAE.append(acc_MAE)
        peakdist, peakdistMEAN = peak_disterr_metric2(prediction.to('cpu'), y_batch.to('cpu'))
        batch_peakdist.append(peakdistMEAN)
        
        if train:
            wandb.log({'train_err_MSE':acc_MSE})
            wandb.log({'train_err_MAE':acc_MAE})
            wandb.log({'train_peakDistErrMEAN': peakdistMEAN})
            wandb.log({'train_peakDistErr': peakdist})
            #print(f"errors logged")
        else:
            wandb.log({'val_err_MSE':acc_MSE})
            wandb.log({'val_err_MAE':acc_MAE})
            wandb.log({'val_peakDistErrMEAN': peakdistMEAN})
            wandb.log({'val_peakDistErr': peakdist})
            

    if sizeBatch ==0 or numBatch == 0:
        print(f" sizeBatch: {sizeBatch}   numBatch:  {numBatch}")
    sizeBatch = sizeBatch / numBatch # get the average batch size

    if len(batch_acc_MSE) != numBatch:
        print(f"You're maths logic was faulty!", numBatch, len(batch_acc_MSE))
        print(batch_acc_MSE)

    batch_acc_MSE_mean = (sum(batch_acc_MSE) / len(batch_acc_MSE))
    batch_acc_MAE_mean = (sum(batch_acc_MAE) / len(batch_acc_MAE))
    batch_peakdist_mean = (sum(batch_peakdist) / len(batch_peakdist))
    print(f"peak dist means  {batch_peakdist_mean}")
    accs = {'baseAcc': acc[2],'MSE':batch_acc_MSE_mean, 'MAE':batch_acc_MAE_mean, 'peakDist':batch_peakdist_mean}
    
    if train:
        return current_loss, predict_list, labels, accs, model, optimizer, img_batch, imNorm_batch #, lr_ls
    else:
        return current_loss, predict_list, labels, accs, img_batch, imNorm_batch # changed y_batch to labels in return 


#                   model,test, loss_fn, config.batch_size, device, config
def test_loop_batch(model,data, loss_fn, batch_size, device, config, runname="", save_loc =""):
    import sys
    from plottingP3Direction import plot_predictions
    sys.path.append('../.')
    model = model.eval()
    predict_list = []
    label_list = []
    peakdists = []
    total_count =0
    num_correct = 0
    correct = 0
    baseacc_list = []
    MSE_list = []
    MAE_list = []
    peakdist_list = []

    with torch.no_grad():
        for i, batch in enumerate(data,0):
            tense, label, img_batch, imNorm_batch = batch #, img_batch, imNorm_batch
            
            prediction = model.forward(tense.to(device))
            """for i in range(len(label)-1):
                #print(len(label), label[0].argmax(), len(label)-1)
                if label[i].argmax() == prediction[i].argmax():
                    num_correct +=1"""

            tacc = get_roughAcc(config.std_dev, label, prediction)
            wandb.log({"Test acc rough": tacc[2]})
            #[predict_list.append(pred.argmax().item()) for pred in prediction]
            [predict_list.append(pred.argmax().to('cpu')) for pred in prediction]  #.argmax()  # .argmax(),.item(),.argmax(),.item()
            [label_list.append(lab.argmax().to('cpu')) for lab in label] #.argmax()  # .argmax(),.item(),.argmax(),.item()
            
            total_count += batch_size

       
        
            test_err_MSE = MSE_metric(prediction.to('cpu'), label.to('cpu'))
            test_err_MAE =  MAE_metric(prediction.to('cpu'), label.to('cpu'))
            test_peakdist, testpeakdistMEAN = peak_disterr_metric2(prediction.to('cpu'), label.to('cpu'))
            wandb.log({'test_acc_MSE':test_err_MSE})
            wandb.log({'test_acc_MAE':test_err_MAE})
            wandb.log({'test_peakDistErr': test_peakdist})
            wandb.log({'test_peakDist': testpeakdistMEAN})
            MSE_list.append(test_err_MSE)
            MAE_list.append(test_err_MAE)
            peakdist_list.append(testpeakdistMEAN)
            baseacc_list.append(tacc[2])

        

        plot_predictions(prediction, label, test_peakdist, runname=runname, save_loc=save_loc) # compare label and prediction distribution/ num_samples=len(tense),

        #print('test accuracy MSE: ', test_acc_MSE )
        #print('test accuracy MAE: ', test_acc_MAE)
        #print(f"test accuracy peak dist  err {test_peakdist}")
        
        
        

        accuracy = {'BaseAcc':baseacc_list,'MSE': MSE_list, 'MAE': MAE_list, 'peakDist': peakdist_list}
        return accuracy, predict_list, label_list

def train_val_batch(model, train, val, save_dict, lr, loss_fn, epochs, batch_size, optimizer, scheduler, device, config): #train_dl, val_dl, 
    #print("Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 ** 3) 
    import sys
    sys.path.append('../.')
    import pickle
    #import wandb
    from IPython.display import clear_output
    IP = ImageProcessor(device)

    # to reduce number of var changes later on
    x_train, resolution, av_lum, model_name, half_ciprange, std_dev, batchsize = train
    x_val  = val
    
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

    """if optimizer: # need a choose scheduler function!
        print("Optimizer present: ",optimizer)
        scheduler = choose_scheduler(save_dict, optimizer)"""
        
    for epoch in tqdm(range(save_dict['start_epoch'],epochs)):
        print(f"Data Loading...")
        train_ds = IDSWDataSetLoader3(x_train, resolution, av_lum, model_name, half_ciprange, std_dev, device)# av_lum, res,pad,
        trainL = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True) #, num_workers=2

        val_ds= IDSWDataSetLoader3(x_val, resolution, av_lum, model_name, half_ciprange, std_dev, device)
        val = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        random_value = random.randrange(0,batch_size)
        print('Training...')
        # , img_batch, imNorm_batch
        # if i initialise the tran and val dataloaders here - I would get different augmented images each epoch but with the same base images
        # i could tie this with more epochs to overall have a larger ds

        t_loss, train_prediction, t_label_list, tacc, model, optimizer, img_batch, imNorm_batch = loop_batch(model, 
                                                                                                             trainL,
                                                                                                             loss_fn,
                                                                                                             batch_size,
                                                                                                             sample, 
                                                                                                             random_value, 
                                                                                                             epoch,  
                                                                                                             IP,
                                                                                                             save_dict, 
                                                                                                             device, config,
                                                                                                             optimizer, 
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

        print('Validating...')

        v_loss, val_prediction, v_label_list, vacc, img_batch, imNorm_batch = loop_batch(model, 
                                                                                            val, 
                                                                                            loss_fn,
                                                                                            batch_size,
                                                                                            sample,
                                                                                            random_value,
                                                                                            epoch, 
                                                                                            IP,
                                                                                            save_dict, 
                                                                                            device, config,
                                                                                            optimizer = None, 
                                                                                            train = False)
        print(f"v loss  {v_loss}")
        if scheduler and scheduler is not "NoSched":
            scheduler.step(v_loss)
            #print("sched step")
            #print(scheduler.state_dict())
            
        v_loss_list.append(v_loss)
        #[v_predict_list.append(pred) for pred in val_prediction]
        v_predict_list.append(val_prediction)
        wandb.log({'v_loss':v_loss})

        v_accuracy_list.append(vacc)


        wandb.log({'c_epoch':int(epoch)})
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

