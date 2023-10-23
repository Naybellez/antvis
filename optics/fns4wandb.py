# fucntions for compatibility with WANB

import cv2
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as maths

import os
import random

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.nn import functional
#from torchsummary import summary
#import torchvision.transforms as transforms

from tqdm import tqdm
from IPython.display import clear_output
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import wandb
import pprint

from loop_fns import loop, test_loop
from functions import import_imagedata, get_data, label_oh_tf,  Unwrap, ImageProcessor

from architectures import vgg16net, smallnet1, smallnet2, smallnet3, build_net


#                    OPTIMISERS

def build_optimizer(network, optimizer, learning_rate, weight_decay=0):
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        if weight_decay == 0:
            optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer



def set_optimizer(optim):
	optim_list=[]
	if optim =='Adam':
		optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
		optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
		optim_list.append(optimizer1)
		optim_list.append(optimizer2)
	elif optim == 'SGD':
		optimizer3 = torch.optim.SGD(model.parameters(), lr=learning_rate)
		optim_list.append(optimizer3)
	return optim_list


def set_lossfn(lf):
    if lf =='MSE':
        loss_fn = nn.MSELoss()
    elif lf == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn

def choose_model(config):
    if config.model_name == 'build_net':
        return build_net(config.lin_layer_size,config.dropout, config.first_lin_lay, config.kernal_size, config.channel_num)
    elif config.model_name == 'smallnet1':
        return smallnet1(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks= config.ks)
    elif config.model_name == 'smallnet2':
        return smallnet2(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks = config.ks)
    elif config.model_name == 'smallnet3':
        return smallnet3(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks=config.ks)
    elif config.model_name == 'vgg16net':
        return vgg16net(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks=config.ks, dropout= config.dropout)
    else:
        print('Model Name Not Recognised')

#           PIPLINE FUNCTIONS

                                # HP Sweep
def train(config, col_dict):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path= config.image_path)

    model = choose_model(config).to(device)
    loss_fn = set_lossfn(config.loss_fn)
    
    e_count = 0
    
    
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler, last_epoch=-1)
    
    for epoch in range(config.epochs):

        t_loss, predict_list, t_num_correct, model, optimizer = loop(model, x_train, y_train, epoch, loss_fn, device, col_dict, num_classes=config.num_classes, optimizer=optimizer)
        t_accuracy = (t_num_correct /len(x_train))*100
        v_loss, __, v_num_correct= loop(model, x_val, y_val, epoch, loss_fn, device,col_dict, num_classes= config.num_classes,train=False) 
        v_accuracy= (v_num_correct / len(x_val))*100
        
        t_avg_loss =t_loss/len(x_train)
        v_avg_loss = v_loss /len(x_val)
        
        e_count +=1
        # logging
        wandb.log({'avg_train_loss': t_avg_loss, 'epoch':epoch})
        wandb.log({'avg_val_loss': v_avg_loss, 'epoch':epoch})
        wandb.log({'train_loss': t_loss, 'epoch':epoch})
        wandb.log({'val_loss': v_loss, 'epoch':epoch})
        wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
        wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})
    return model

                                #Training
            
def train_model(model, x_train, y_train, x_val, y_val,loss_fn, config, col_dict,  device): # training
    wandb.watch(model, loss_fn, log='all', log_freq=10)
    
    sample_count =0
    batch_count = 0
    e_count = 0

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler, last_epoch=-1)
    
    for epoch in tqdm(range(config.epochs)):            
        #train                                                                  
        t_loss, predict_list, t_num_correct, model, optimizer = loop(model, x_train, y_train, epoch, loss_fn, device, col_dict, config.num_classes, optimizer=optimizer, scheduler=scheduler)
        sample_count += len(x_train)
       
        # validation
        v_loss, __, v_num_correct= loop(model, x_val, y_val, epoch, loss_fn, device,col_dict, config.num_classes, train=False) 
        batch_count +=1
        
        if (batch_count +1)%2 ==0:
            train_log(t_loss,v_loss, sample_count, epoch)
        e_count +=1
        clear_output()
        
    
def pipeline(config, col_dict, title, image_file_path):
	device = "cuda:1" if torch.cuda.is_available() else "cpu"
	x_train, y_train, x_val, y_val, x_test,y_test = get_data(file_path=image_file_path)
	with wandb.init(project=title, config=config):
		config = wandb.config
		print(col_dict)
		print(config.model_name)
		model = choose_model(config).to(device)
		loss_fn = set_lossfn(config.loss_fn)
		train_model(model, x_train, y_train, x_val, y_val, loss_fn, config, col_dict, device)
		test_loop(model, x_text, y_test, device, col_dict, title)
	return model



#                                LOGGING 


def train_log(t_loss, v_loss, sample_count, epoch):
    wandb.log({'epoch': epoch,
              't_loss': t_loss,
              'v_loss': v_loss},
             step=sample_count)
    print(f'loss after {str(sample_count).zfill(5)} examples: {v_loss:.3f}')



def log_test_score(correct, accuracy, X):

	wandb.log({'Test_accuracy %':accuracy})
	wandb.log({'test accuracy #': correct})
	torch.onnx.export(model, X, f'{title}_accuracy{accuracy}.onnx')
	wandb.save(f'{title}_{accuracy}.onnx')
