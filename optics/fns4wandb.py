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

#import wandb
import pprint

from loop_fns import loop, test_loop
from functions import import_imagedata, get_data, label_oh_tf,  Unwrap, ImageProcessor

#from architectures import sevennet, smallnet1, smallnet2, smallnet3, build_net
from copy import deepcopy
import pickle
from torch.utils.data import DataLoader, Dataset


#                    OPTIMISERS

def build_optimizer(network, optimizer, learning_rate, weight_decay=0):
    if optimizer == 'SGD':

        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        #if weight_decay == 0:
            
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate, weight_decay=weight_decay)
    return optimizer



def set_optimizer(optim, model, learning_rate):
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
    #print('c5', type(config))
    #print(config.model_name)
    for model_card in config.model_cards:
        if model_card['name'] == '4c3l':
            return smallnet1(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks= config.ks)
        elif model_card['name'] == '3c2l':
            return smallnet2(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks = config.ks)
        elif model_card['name'] == '2c2l':
            return smallnet3(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks=config.ks)
        elif model_card['name'] == '7c3l':
            return sevennet(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks=config.ks, dropout= config.dropout)
        elif model_card['name'] == 'vgg16':
            model_vgg16 = vgg16(weights="IMAGENET1K_V1")
            vgg_feats = model_vgg16.features
            vgg_classifier = model_vgg16.classifier
            vgg_classifier.pop(6)
    
            vgg = nn.Sequential(
                vgg_feats,
                Flattern(),
                vgg_classifier,
                nn.Linear(4096,11),
                nn.Softmax(dim=0),
                )
            return vgg
        else:
            print('Model Name Not Recognised')
"""
def choose_model(config):
    #print('c5', type(config))
    #print(config.model_name)
    if config.model_name == 'build_net':
        return build_net(config.lin_layer_size,config.dropout, config.first_lin_lay, config.kernal_size, config.channel_num)
    elif config.model_name == '4c3l':
        return smallnet1(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks= config.ks)
    elif config.model_name == '3c2l':
        return smallnet2(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks = config.ks)
    elif config.model_name == '2c2l':
        return smallnet3(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks=config.ks)
    elif config.model_name == '7c3l':
        return sevennet(in_chan=config.channels, f_lin_lay=config.first_lin_lay, l_lin_lay=config.num_classes, ks=config.ks, dropout= config.dropout)
    elif config.model_name == 'vgg16':
        model_vgg16 = vgg16(weights="IMAGENET1K_V1")
        vgg_feats = model_vgg16.features
        vgg_classifier = model_vgg16.classifier
        vgg_classifier.pop(6)

        vgg = nn.Sequential(
            vgg_feats,
            Flattern(),
            vgg_classifier,
            nn.Linear(4096,11),
            nn.Softmax(dim=0),
            )
        return vgg
    else:
        print('Model Name Not Recognised')


"""
#           PIPLINE FUNCTIONS
#from loop_fns import print_gpu_mem
                                # HP Sweep
def hp_sweep(config, col_dict,save_dict, device,seed,model, loss_fn, optimizer, scheduler, best_acc=0, data=None):
    #if data==None:
    #    x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path= config.image_path, seed=seed)
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path= config.image_path, seed=seed)
    #elif isinstance(data, tuple):
    #    x_train, y_train, x_val, y_val, x_test, y_test = data
    #else:
    #    print('Hungry for Data')

    #model = choose_model(config).to(device)
    #loss_fn = set_lossfn(config.loss_fn)
    #optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    #scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler, last_epoch=-1)


    e_count = 0
    t_loss_list = []
    v_loss_list = []#
    t_predict_list = []
    v_predict_list = []#
    t_accuracy_list = []#
    v_accuracy_list = []
    t_label_list = []#
    v_label_list = []
    #prepro = ImageProcessor(device)
    
    
    for epoch in tqdm(range(config.epochs)):

        t_correct =0
        v_correct =0
        #print('Katara')
        #print_gpu_mem()

        #if epoch == 0:
        #    model = model.to('cpu')
        #    best_model = deepcopy(model)
        #    model = model.to(device)
        
        model.train()
        print('training...')
        #print('Sokka')
        #print_gpu_mem()
      
        t_loss, t_predictions, t_num_correct,t_labels, model, optimizer = loop(model, x_train[:10], y_train[:10], epoch, loss_fn, device, col_dict, num_classes=config.num_classes, pad_size=col_dict['padding'],optimizer=optimizer, scheduler=scheduler)
        #print('Zuko')
        #print_gpu_mem()
        t_accuracy = (t_num_correct /len(x_train))*100
        t_accuracy_list.append(t_accuracy)
        #[t_label_list.append(i.to('cpu')) for i in t_labels]
        t_label_list.append(t_labels)
        #[t_predict_list.append(i.to('cpu')) for i in t_predictions]
        t_predict_list.append(t_predictions)
        #[t_loss_list.append(i.to('cpu')) for i in t_losses]
        t_loss_list.append(t_loss)
        #print('Azula')
        #print_gpu_mem()
        model.eval()
        print('validating...') #current_loss, predict_list, num_correct
        v_loss, v_predictions, v_num_correct, v_labels= loop(model, x_val[:10], y_val[:10], epoch, loss_fn, device,col_dict, num_classes= config.num_classes, train=False) 
        v_accuracy= (v_num_correct / len(x_val))*100
        #print('Toph')
        #print_gpu_mem()
        #v_accuracy_list.append(v_accuracy)
        #[v_label_list.append(i.to('cpu')) for i in v_labels]
        v_label_list.append(v_labels)
        #[v_predict_list.append(i.to('cpu')) for i in v_predictions]
        v_predict_list.append(v_predictions)
        v_loss_list.append(v_loss)
        #[v_loss_list.append(i.to('cpu')) for i in v_losses]
        #t_avg_loss =t_loss/len(x_train)
        #v_avg_loss = v_loss /len(x_val)
        
        
        # logging
        #wandb.log({'avg_train_loss': t_avg_loss, 'epoch':epoch})
        #wandb.log({'avg_val_loss': v_avg_loss, 'epoch':epoch})
        wandb.log({'train_loss': t_loss, 'epoch':epoch})
        wandb.log({'val_loss': v_loss, 'epoch':epoch})
        wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
        wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})

        save_dict['Current_Epoch'] = config['epochs']
        save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
        save_dict['validation_samples'] = len(x_val)
        save_dict['t_loss_list'] = t_loss_list #[c.to('cpu') for c in t_loss_list]
        save_dict['t_predict_list'] = [[c.to('cpu') for c in k]for k in t_predict_list] #[[c.to('cpu') for c in k]for k in t_predict_list]  # [c.to('cpu') for c in t_predict_list] 
        save_dict['t_accuracy_list'] = t_accuracy_list #
        save_dict['v_loss_list'] = v_loss_list #[c.to('cpu') for c in v_loss_list]
        save_dict['v_predict_list'] = [[c.to('cpu') for c in k]for k in v_predict_list]#[[c.to('cpu') for c in k]for k in v_predict_list] # [c.to('cpu') for c in v_predict_list]
        save_dict['v_accuracy_list'] = v_accuracy_list #
        save_dict['t_labels'] = [[c.to('cpu') for c in k]for k in t_label_list]
        save_dict['v_labels'] = [[c.to('cpu') for c in k] for k in v_label_list]

        
        #print('suki')
        #print_gpu_mem()

        """if v_accuracy > best_acc:
            best_acc = v_accuracy
            #model = model.to('cpu')
            best_model = deepcopy(model)
            #model = model.to(device)
            best_model.to('cpu')
            #print(best_model.state_dict(), '\n', type(best_model.state_dict()))
            save_dict['Current_Epoch'] = config['epochs']
            save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
            save_dict['validation_samples'] = len(x_val)
            save_dict['t_loss_list'] = [c.to('cpu') for c in t_loss_list]
            save_dict['t_predict_list'] = [c.to('cpu') for c in t_predict_list]  
            save_dict['t_accuracy_list'] = t_accuracy_list #
            save_dict['v_loss_list'] = [c.to('cpu') for c in v_loss_list]
            save_dict['v_predict_list'] = [c.to('cpu') for c in v_predict_list] #
            save_dict['v_accuracy_list'] = v_accuracy_list #
            save_dict['t_labels'] = [c.to('cpu') for c in t_label_list]
            save_dict['v_labels'] = [c.to('cpu') for c in v_label_list]
            #save_dict['model.state_dict'] = {keys:values.to('cpu') for keys, values in best_model.state_dict().items()}
            #save_dict['model_architecture_untrained'] = model_architecture

            title = save_dict['Run']
            with open(f"{save_dict['save_location']}{title}.pkl", 'wb+') as f:
                pickle.dump(save_dict, f)
            
            print('improvment in metrics. model saved')"""


        e_count +=1

    return model, save_dict, x_test, y_test

def hp_sweep_DL(config, col_dict,save_dict, device,seed,model,best_acc=0, data=None):
    
    file_path = r'//smbhome.uscs.susx.ac.uk/nn268/Documents/PHD/antvis/optics/AugmentedDS_IDSW'
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path, seed=8)
    train_loader = IDSWDataSetLoader(x_train, y_train, col_dict=col_dict, device=device)
    train_loader = DataLoader(train_loader, shuffle=False, batch_size=4)
    val_loader = IDSWDataSetLoader(x_val, y_val, col_dict=col_dict, device=device)
    val_loader = DataLoader(val_loader, shuffle=False, batch_size=4)
    test_loader = IDSWDataSetLoader(x_test, y_test, col_dict=col_dict, device=device)
    test_loader = DataLoader(test_loader, shuffle=False, batch_size=4)

    #model = choose_model(config).to(device)
    
    if config.loss_fn == 'MSE':
            loss_fn = nn.MSELoss()
    elif config.loss_fn == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler, last_epoch=-1)


    e_count = 0
    t_loss_list = []
    v_loss_list = []#
    t_predict_list = []
    v_predict_list = []#
    t_accuracy_list = []#
    v_accuracy_list = []
    t_label_list = []#
    v_label_list = []
    #prepro = ImageProcessor(device)
    
    
    for epoch in tqdm(range(config.epochs)):

        t_correct =0
        v_correct =0

        model.train()
        print('training...')

      
        t_loss, t_predictions, t_num_correct,t_labels, model, optimizer = batch_loop(model, train_loader, epoch, loss_fn, device, col_dict, num_classes=config.num_classes, pad_size=col_dict['padding'],optimizer=optimizer, scheduler=scheduler)

        t_accuracy = (t_num_correct /len(x_train))*100
        t_accuracy_list.append(t_accuracy)
        t_label_list.append(t_labels)
        t_predict_list.append(t_predictions)
        t_loss_list.append(t_loss.item())
        
        model.eval()
        print('validating...') #current_loss, predict_list, num_correct
        v_loss, v_predictions, v_num_correct, v_labels= batch_loop(model, val_loader, epoch, loss_fn, device,col_dict, num_classes= config.num_classes, train=False) 
        
        v_accuracy= (v_num_correct / len(x_val))*100
        v_accuracy_list.append(v_accuracy)
        v_label_list.append(v_labels)
        v_predict_list.append(v_predictions)
        v_loss_list.append(v_loss.item())

        
        
        # logging
        #t_avg_loss =t_loss/len(x_train)
        #v_avg_loss = v_loss /len(x_val)

        e_count +=1
        # logging
        #wandb.log({'avg_train_loss': t_avg_loss, 'epoch':epoch})
        #wandb.log({'avg_val_loss': v_avg_loss, 'epoch':epoch})

        wandb.log({'train_loss': t_loss, 'epoch':epoch})
        wandb.log({'val_loss': v_loss, 'epoch':epoch})

        wandb.log({'train_correct': t_num_correct, 'epoch':epoch})
        wandb.log({'val_correct': v_num_correct, 'epoch':epoch})

        wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
        wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})

        wandb.log({'t_labels': t_label_list, 'epoch':epoch})
        wandb.log({'v_labels': v_label_list, 'epoch':epoch})

        wandb.log({'t_predictions': t_predict_list, 'epoch':epoch})
        wandb.log({'v_predictions': v_predict_list, 'epoch':epoch})

            # add lists to save dict after all epochs run
        save_dict['Current_Epoch'] = config['epochs']
        save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
        save_dict['validation_samples'] = len(x_val)
        save_dict['t_loss_list'] = t_loss_list #[c.to('cpu') for c in t_loss_list]
        save_dict['t_predict_list'] = [[c.to('cpu') for c in k]for k in t_predict_list] #[[c.to('cpu') for c in k]for k in t_predict_list]  # [c.to('cpu') for c in t_predict_list] 
        save_dict['t_accuracy_list'] = t_accuracy_list #
        save_dict['v_loss_list'] = v_loss_list #[c.to('cpu') for c in v_loss_list]
        save_dict['v_predict_list'] = [[c.to('cpu') for c in k]for k in v_predict_list]#[[c.to('cpu') for c in k]for k in v_predict_list] # [c.to('cpu') for c in v_predict_list]
        save_dict['v_accuracy_list'] = v_accuracy_list #
        save_dict['t_labels'] = t_label_list #[[c.to('cpu') for c in k]for k in t_label_list]
        save_dict['v_labels'] = v_label_list #[[c.to('cpu') for c in k] for k in v_label_list]

        e_count +=1
        
    title = save_dict['Run']
    # TESTING

    test_predictions, test_y, test_accuracy = test_loop_batch(model, test_loader, loss_fn, device, col_dict, title, config.num_classes)
    save_dict['test_predictions']= [c.to('cpu') for c in test_predictions]
    save_dict['test_labels'] = test_y
    save_dict['test_acc'] = test_accuracy
    
    with open(f"/its/home/nn268/antvis/antvis/optics/pickles/{title}.pkl", 'wb+') as f:
        pickle.dump(save_dict, f)

    return model, save_dict


                                #Training
            
def train_model(model, x_train, y_train, x_val, y_val,loss_fn, config, col_dict, save_dict, device): # training. model, x_train, y_train, x_val, y_val,loss_fn, config, col_dict,  device #model, train_loader, val_loader,loss_fn, config, col_dict,  device)
    wandb.watch(model, loss_fn, log='all', log_freq=10)
    #print('train model, col dict is a', type(col_dict) )
    sample_count =0  # *
    batch_count = 0  # *
    e_count = 0
    # lists for save dict
    t_loss_list = []
    v_loss_list =[]
    t_predict_list = []
    t_label_list = []
    v_predict_list = []
    v_label_list = []
    t_accuracy_list= []
    v_accuracy_list= []

    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler, last_epoch=-1)
    
    for epoch in tqdm(range(config.epochs)):            
        #train                                                      
        #print('pre loop, col_dict is a: ', type(col_dict))   
        # current_loss, predict_list, num_correct, label_list,loss_list, model, optimizer   #model, loader, epoch, loss_fn, device, col_dict, num_classes, pad_size =5, optimizer =None, scheduler= None, train =True                                    
        t_loss, t_predict_list_, t_num_correct, t_label_list_, model, optimizer = loop(model, x_train, y_train, epoch, loss_fn, device, col_dict, config.num_classes, optimizer=optimizer, scheduler=scheduler) #model, x_train, y_train, epoch, loss_fn, device, col_dict, config.num_classes, optimizer=optimizer, scheduler=scheduler
        sample_count += len(x_train)  # *
        # * accuracy = (num_correct/len(x_train))*100
        print('train loss: ', t_loss)
        t_loss_list.append(t_loss)
        t_predict_list.append(t_predict_list_)
        t_label_list.append(t_label_list_)
        t_accuracy_list.append(t_num_correct/len(x_train))
       
        # validation        #current_loss, predict_list, num_correct, label_list,loss_list
        v_loss, v_predict_list_, v_num_correct, v_label_list_= loop(model, x_val,y_val, epoch, loss_fn, device,col_dict, config.num_classes, train=False) 
        print('v loss', v_loss)
        # * accuracy = (num_correct/len(x_val))*100
        v_loss_list.append(v_loss)
        v_predict_list.append(v_predict_list_)
        v_label_list.append(v_label_list_)
        v_accuracy_list.append(v_num_correct/len(x_val))

        batch_count +=1
        # * e_countt +=1
        # wandb logging
        wandb.log({'train_loss': t_loss, 'epoch':epoch})
        wandb.log({'val_loss': v_loss, 'epoch':epoch})

        wandb.log({'train_correct': t_num_correct, 'epoch':epoch})
        wandb.log({'val_correct': v_num_correct, 'epoch':epoch})
        
        t_accuracy = (t_num_correct/len(x_train))*100
        v_accuracy = (v_num_correct/len(x_val))*100
        wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
        wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})

        wandb.log({'t_labels': t_label_list})#, 'epoch':epoch})
        wandb.log({'v_labels': v_label_list})#, 'epoch':epoch})

        wandb.log({'t_predictions': t_predict_list})#, 'epoch':epoch})
        wandb.log({'v_predictions': v_predict_list})#, 'epoch':epoch})
        #if (batch_count +1)%2 ==0:
        #    #train_log(t_loss_list,v_loss_list, epoch) #t_loss, v_loss, epoch)
           
        e_count +=1

        

        #clear_output()
    #print('t loss type: ',type(t_loss_list))
    #print('predict list type: ',type(t_predict_list))
    #print('t label list type: ',type(t_label_list))
    #print('v loss type: ',type(v_loss_list))
    #print('v predict list type: ',type(v_predict_list))
    #print('v label list type: ',type(v_label_list))


    #print('t predictions', len(t_predict_list), t_predict_list, '\n')
    #print('v predictions', len(v_predict_list),v_predict_list,  '\n')
    #print('t labels ', len(t_label_list),t_label_list, '\n')
    #print('v labels', len(v_label_list),v_label_list, '\n')
    #print('t accuracy ',len(t_accuracy_list),t_accuracy_list, '\n')
    #print('t loss list: ', t_loss_list)
    #print('v loss list: ', v_loss_list)

    # add lists to save dict after all epochs run
    save_dict['Current_Epoch'] = config['epochs']
    save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
    save_dict['validation_samples'] = len(x_val)
    save_dict['t_loss_list'] = t_loss_list #[c.to('cpu') for c in t_loss_list]
    save_dict['t_predict_list'] = [[c.to('cpu') for c in k]for k in t_predict_list] #[[c.to('cpu') for c in k]for k in t_predict_list]  # [c.to('cpu') for c in t_predict_list] 
    save_dict['t_accuracy_list'] = t_accuracy_list #
    save_dict['v_loss_list'] = v_loss_list #[c.to('cpu') for c in v_loss_list]
    save_dict['v_predict_list'] = [[c.to('cpu') for c in k]for k in v_predict_list]#[[c.to('cpu') for c in k]for k in v_predict_list] # [c.to('cpu') for c in v_predict_list]
    save_dict['v_accuracy_list'] = v_accuracy_list #
    save_dict['t_labels'] = [[c.to('cpu') for c in k]for k in t_label_list]
    save_dict['v_labels'] = [[c.to('cpu') for c in k] for k in v_label_list]

    title = save_dict['Run']
    with open(f"/its/home/nn268/antvis/antvis/optics/pickles/{title}.pkl", 'wb+') as f:
        pickle.dump(save_dict, f)

    #print(save_dict)

    return save_dict


#edits to include a dataloader 10/01/24
from functions import IDSWDataSetLoader
from functions import get_data
from functions import ImageProcessor

def pipeline(config, col_dict,save_dict, title, device, seed):
    #print('c3',config)
    #device = "cuda:1" if torch.cuda.is_available() else "cpu"
    #print('pipeline, config is type: ',type(config))
    #print(config.keys())
    #print('pipeline, col_dict is a: ',type(col_dict))
    #print(col_dict.keys())
    #loader = IDSWDataSetLoader(col_dict,device)
    #images, labels = loader[:]
    #random_seed = random.seed(8)

    # data processing
    #x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=random_seed, shuffle=True)
    #i = random.randrange(len(x_train))
    #IP = ImageProcessor(device)
    #IP.view(x_train[i], scale =5)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, random_state=random_seed, shuffle=True)
    #IP.view(x_train[i], scale =5)
    
    #train_loader = DataLoader(list(zip(x_train,y_train)), batch_size=16,shuffle=True)
    #val_loader = DataLoader(list(zip(x_val,y_val)), batch_size=16,shuffle=True)
    #test_loader = DataLoader(list(zip(x_test,y_test)), batch_size=16,shuffle=True)

    

    with wandb.init(project=title, config=config):
        config = wandb.config
        # * loading in data
        x_train, y_train, x_val, y_val, x_test, y_test = get_data(r'/its/home/nn268/antvis/antvis/optics/AugmentedDS_IDSW/', seed)  # *

        #print('c4', type(config),config)
        #print(col_dict)
        #print(config.model_name)
        model = choose_model(config).to(device) ###
        loss_fn = set_lossfn(config.loss_fn)

        train_model(model, x_train, y_train, x_val, y_val, loss_fn, config, col_dict, save_dict, device)
        #t_save_dict = train_model(model, train_loader, val_loader, loss_fn, config, col_dict, device)
        test_loop(model, x_test, y_test, loss_fn, device, col_dict, title, config.num_classes) #(model, X, Y, loss_fn, device, col_dict,title, num_classes)
        #train_model(model, x_train, y_train, x_val, y_val, loss_fn, config, col_dict, device)
        #test_loop(model, x_text, y_test, device, col_dict, title)

        

    return model

# below works!!!!! 290124
def train(device,col_dict, save_dict, config=None):
    # lists for save dict
    t_loss_list = []
    v_loss_list =[]
    t_predict_list = []
    t_label_list = []
    v_predict_list = []
    v_label_list = []
    t_accuracy_list= []
    v_accuracy_list= []
    
    with wandb.init(config=config):
        config = wandb.config

        x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path= r'/its/home/nn268/antvis/antvis/optics/AugmentedDS_IDSW/', seed= random.randint(0, 50))
        
        #model =smallnet3(in_chan=3, f_lin_lay=67968, l_lin_lay=11, ks=(3,5)).to(device) #10368
        model = choose_model(config).to(device)
        
        if config.loss_fn == 'MSE':
            loss_fn = nn.MSELoss()
        elif config.loss_fn == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()

        e_count = 0
         # *

        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

        for epoch in tqdm(range(config.epochs)):
            # current_loss, predict_list, num_correct, label_list, model, optimizer
            t_loss, t_predict_list_, t_num_correct, t_label_list_, model, optimizer = loop(model, x_train, y_train, epoch, loss_fn, device, col_dict, num_classes=11, optimizer=optimizer)
            t_accuracy = (t_num_correct /len(x_train))*100
            t_loss_list.append(t_loss)
            t_predict_list.append(t_predict_list_)
            t_label_list.append(t_label_list_)
            t_accuracy_list.append(t_accuracy)

            v_loss, v_predict_list_, v_num_correct, v_label_list_= loop(model, x_val, y_val, epoch, loss_fn, device,col_dict,num_classes=11, train=False)
            v_accuracy= (v_num_correct / len(x_val))*100
            v_loss_list.append(v_loss)
            v_predict_list.append(v_predict_list_)
            v_label_list.append(v_label_list_)
            v_accuracy_list.append(v_accuracy)

            t_avg_loss =t_loss/len(x_train)
            v_avg_loss = v_loss /len(x_val)

            e_count +=1
            # logging
            wandb.log({'avg_train_loss': t_avg_loss, 'epoch':epoch})
            wandb.log({'avg_val_loss': v_avg_loss, 'epoch':epoch})

            wandb.log({'train_loss': t_loss, 'epoch':epoch})
            wandb.log({'val_loss': v_loss, 'epoch':epoch})

            wandb.log({'train_correct': t_num_correct, 'epoch':epoch})
            wandb.log({'val_correct': v_num_correct, 'epoch':epoch})

            wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
            wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})

            wandb.log({'t_labels': t_label_list, 'epoch':epoch})
            wandb.log({'v_labels': v_label_list, 'epoch':epoch})

            wandb.log({'t_predictions': t_predict_list, 'epoch':epoch})
            wandb.log({'v_predictions': v_predict_list, 'epoch':epoch})

            # add lists to save dict after all epochs run
    save_dict['Current_Epoch'] = config['epochs']
    save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
    save_dict['validation_samples'] = len(x_val)
    save_dict['t_loss_list'] = t_loss_list #[c.to('cpu') for c in t_loss_list]
    save_dict['t_predict_list'] = [[c.to('cpu') for c in k]for k in t_predict_list] #[[c.to('cpu') for c in k]for k in t_predict_list]  # [c.to('cpu') for c in t_predict_list] 
    save_dict['t_accuracy_list'] = t_accuracy_list #
    save_dict['v_loss_list'] = v_loss_list #[c.to('cpu') for c in v_loss_list]
    save_dict['v_predict_list'] = [[c.to('cpu') for c in k]for k in v_predict_list]#[[c.to('cpu') for c in k]for k in v_predict_list] # [c.to('cpu') for c in v_predict_list]
    save_dict['v_accuracy_list'] = v_accuracy_list #
    save_dict['t_labels'] = t_label_list #[[c.to('cpu') for c in k]for k in t_label_list]
    save_dict['v_labels'] = v_label_list #[[c.to('cpu') for c in k] for k in v_label_list]
    
    title = save_dict['Run']
    test_predictions, test_y, test_accuracy = test_loop(model, x_test, y_test, loss_fn, device, col_dict, title, config.num_classes)
    save_dict['test_predictions']= [c.to('cpu') for c in test_predictions]
    save_dict['test_labels'] = test_y
    save_dict['test_acc'] = test_accuracy

    
    with open(f"/its/home/nn268/antvis/antvis/optics/pickles/{title}.pkl", 'wb+') as f:
        pickle.dump(save_dict, f)
        
    return model


# train with dataloader
from functions import IDSWDataSetLoader
def train_DL(device,col_dict, save_dict, config=None):
    # lists for save dict
    t_loss_list = []
    v_loss_list =[]
    t_predict_list = []
    t_label_list = []
    v_predict_list = []
    v_label_list = []
    t_accuracy_list= []
    v_accuracy_list= []
    
    with wandb.init(config=config):
        config = wandb.config

        # import data
        file_path = r'/its/home/nn268/antvis/antvis/optics/AugmentedDS_IDSW/'
        x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path, seed=8)
        train_loader = IDSWDataSetLoader(x_train, y_train, col_dict=col_dict, device=device)
        train_loader = DataLoader(train_loader, shuffle=False, batch_size=4)
        val_loader = IDSWDataSetLoader(x_val, y_val, col_dict=col_dict, device=device)
        val_loader = DataLoader(val_loader, shuffle=False, batch_size=4)
        test_loader = IDSWDataSetLoader(x_test, y_test, col_dict=col_dict, device=device)
        test_loader = DataLoader(test_loader, shuffle=False, batch_size=4)
        #x_train, y_train, x_test, y_test, x_val, y_val = IDSWDataSetLoader(col_dict, device)
        #train_data_loader = DataLoader([x_train, y_train], batch_size=4, shuffle=True)
        #test_data_loader = DataLoader([x_test, y_test], batch_size=4, shuffle=True)
        #val_data_loader = DataLoader([x_val, y_val], batch_size=4, shuffle=True)
        #model =smallnet3(in_chan=3, f_lin_lay=67968, l_lin_lay=11, ks=(3,5)).to(device) #10368
        model = choose_model(config).to(device)
        
        if config.loss_fn == 'MSE':
            loss_fn = nn.MSELoss()
        elif config.loss_fn == 'CrossEntropy':
            loss_fn = nn.CrossEntropyLoss()

        e_count = 0
         # *

        optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

        for epoch in tqdm(range(config.epochs)):
            # current_loss, predict_list, num_correct, label_list, model, optimizer
            t_loss, t_predict_list_, t_num_correct, t_label_list_, model, optimizer = batch_loop(model, train_loader, epoch, loss_fn, device, col_dict, num_classes=11, optimizer=optimizer)
            t_accuracy = (t_num_correct /len(x_train))*100
            t_loss_list.append(t_loss)
            t_predict_list.append(t_predict_list_)
            t_label_list.append(t_label_list_)
            t_accuracy_list.append(t_accuracy)

            v_loss, v_predict_list_, v_num_correct, v_label_list_= batch_loop(model, val_loader, epoch, loss_fn, device,col_dict,num_classes=11, train=False)
            v_accuracy= (v_num_correct / len(x_val))*100
            v_loss_list.append(v_loss)
            v_predict_list.append(v_predict_list_)
            v_label_list.append(v_label_list_)
            v_accuracy_list.append(v_accuracy)

            t_avg_loss =t_loss/len(x_train)
            v_avg_loss = v_loss /len(x_val)

            e_count +=1
            # logging
            wandb.log({'avg_train_loss': t_avg_loss, 'epoch':epoch})
            wandb.log({'avg_val_loss': v_avg_loss, 'epoch':epoch})

            wandb.log({'train_loss': t_loss, 'epoch':epoch})
            wandb.log({'val_loss': v_loss, 'epoch':epoch})

            wandb.log({'train_correct': t_num_correct, 'epoch':epoch})
            wandb.log({'val_correct': v_num_correct, 'epoch':epoch})

            wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
            wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})

            wandb.log({'t_labels': t_label_list, 'epoch':epoch})
            wandb.log({'v_labels': v_label_list, 'epoch':epoch})

            wandb.log({'t_predictions': t_predict_list, 'epoch':epoch})
            wandb.log({'v_predictions': v_predict_list, 'epoch':epoch})

            # add lists to save dict after all epochs run
    save_dict['Current_Epoch'] = config['epochs']
    save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
    save_dict['validation_samples'] = len(x_val)
    save_dict['t_loss_list'] = t_loss_list #[c.to('cpu') for c in t_loss_list]
    save_dict['t_predict_list'] = [[c.to('cpu') for c in k]for k in t_predict_list] #[[c.to('cpu') for c in k]for k in t_predict_list]  # [c.to('cpu') for c in t_predict_list] 
    save_dict['t_accuracy_list'] = t_accuracy_list #
    save_dict['v_loss_list'] = v_loss_list #[c.to('cpu') for c in v_loss_list]
    save_dict['v_predict_list'] = [[c.to('cpu') for c in k]for k in v_predict_list]#[[c.to('cpu') for c in k]for k in v_predict_list] # [c.to('cpu') for c in v_predict_list]
    save_dict['v_accuracy_list'] = v_accuracy_list #
    save_dict['t_labels'] = t_label_list #[[c.to('cpu') for c in k]for k in t_label_list]
    save_dict['v_labels'] = v_label_list #[[c.to('cpu') for c in k] for k in v_label_list]
    
    title = save_dict['Run']
    test_predictions, test_y, test_accuracy = test_loop_batch(model, test_loader, loss_fn, device, col_dict, title, config.num_classes)
    save_dict['test_predictions']= [c.to('cpu') for c in test_predictions]
    save_dict['test_labels'] = test_y
    save_dict['test_acc'] = test_accuracy

    
    with open(f"/its/home/nn268/antvis/antvis/optics/pickles/{title}.pkl", 'wb+') as f:
        pickle.dump(save_dict, f)
        
    return model



#                                LOGGING 


def train_log(t_loss, v_loss, epoch):
    wandb.log({'epoch': epoch,
              't_loss': t_loss,
              'v_loss': v_loss},
             )
    



def log_test_score(correct, accuracy, X):

	wandb.log({'Test_accuracy %':accuracy})
	wandb.log({'test accuracy #': correct})
	torch.onnx.export(model, X, f'{title}_accuracy{accuracy}.onnx')
	wandb.save(f'{title}_{accuracy}.onnx')
