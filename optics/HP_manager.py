import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import DataLoader
#from torch.Utils.data import DataLoader
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

# general maths and image manipulation
import numpy as np
import cv2

# general other
from datetime import date
from tqdm import tqdm
import pprint
import collections
from IPython.display import clear_output
import time
import random


# saving in file types
import csv
import json
import pickle
import os

# simulation logger
import wandb

# my functions
import sys
sys.path.append('../../.')
from functions import  ImageProcessor
from dataPreProcessingP3Direction import get_data
from dataloaderP3Direction import IDSWDataSetLoader3
from fns4wandb import set_lossfn
#from architectures import eightnnet, PrintLayer
from loopsP3Direction import loop_batch, test_loop_batch, train_val_batch
from plotting import learning_curve, accuracy_curve
from plottingP3Direction import plot_confusion
from modelCards import Cards, get_lin_lay, return_card
from modelManagment import choose_model

from fileManagment import save2csv_nest_dict, check_obj4np, save2josn_nested_dict, save2csv, save2json,read_in_json

def getAcc_fromdict(listdict):
    baseacc = []
    MSE = []
    MAE = []
    peak = []
    for item in listdict:
        baseacc.append(item['baseAcc'])
        MSE.append(item['MSE'])
        MAE.append(item['MAE'])
        peak.append(item['peakDist'])
    return baseacc, MSE, MAE, peak



def _go(config=None):
    print(config)
    d = date.today()
    mc = modelcard[0]
    with wandb.init(config=config):  # config=config, project=f"Big Loop batching of model {mc['name']}", notes=f"big loop batcing {mc['name']}.  {d}.",
        config = wandb.config

        for model_idx, model_card in enumerate(config.modelcard): #(config['model_cards']):
            start = time.process_time()
            #print("Current allocated memory (GB):", torch.cuda.memory_allocated() / 1024 ** 3)
            print(model_card)
            model_name = model_card['model']
            model_index = model_card['idx']
            dropout = model_card['dropout'] 
            
            output_lin_lay = 360 ###### Output labels for direction prediction specifically. 
            
            
            for res_idx, resolution_card in enumerate(config.resolutioncard):#config['resolution_cards']):
                resolution = resolution_card['resolution']
                #pad = resolution_card['padding']
                lin_lay = get_lin_lay(model_card, resolution)
                print('lin lay', lin_lay)
            
                #for lr_idx, lr in enumerate(config['learning_rate_cards']):
                scheduler_value = "NoSched"#"no_scheduler"
                print("CONFIG SEEDS :",config.seeds)
                #for seed_idx, seed in enumerate(config.seeds): #enumerate(config['seeds']):
                seed = config.seeds
                #for lossfn_idx, loss in enumerate(config['loss_fn_cards']):
                loss = config.loss_fn_cards
                torch.cuda.empty_cache()

                batch = config.batch_size #['batch_size']

                print('Model: ', str(model_name), f" idx: {model_idx} / {len(config.model_cards)}")
                print('resolution: ', str(resolution), f" idx: {res_idx} / {len(config['resolution_cards'])}")
                #print('learning rate: ', str(lr), f" idx: {lr_idx} / {len(config['learning_rate_cards'])}")
                print('seed: ', str(seed))#, f" idx: {seed_idx} / {len(config['seeds'])}")
                print('loss function: ', str(loss))#, f" idx: {lossfn_idx} / {len(config['loss_fn_cards'])}")
                print('Batch size: ', config.batch_size)#['batch_size'])
                print('Training epochs: ', config.epochs)#['epochs'])
                run_start_time = time.process_time()
                print('start time: ',run_start_time)

                #print(time.process_time() - start)

                epochs = config['epochs'] #40

                IP = ImageProcessor(device)

                wandb.log({'gitHash':gitHASH})
                wandb.log({'Epochs': epochs})

                
                # set save dictionary
                save_dict = {'Run' : f"{model_name}_{resolution}_{d}",
                             'start_epoch' : 0,
                             'Current_Epoch': 0,
                             'save_location' : save_location,
                            'scheduler': scheduler_value}

                model = choose_model(model_name, lin_lay, dropout, output_lin_lay).to(device)
                #model = smallnet3(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout).to(device)

                #print('4')
                #!nvidia-smi
                gauss_range = config.gauss_range
                gauss_width = config.gauss_width


                x_train, y_train, x_val, y_val, x_test, y_test = get_data(seed, "/its/home/nn268/antvis/antvis/optics/NC_IDSW/", locations=[1])
                av_lum = IP.new_luminance(x_train)
                train = (x_train, resolution, av_lum, model_name, config.gauss_range, config.gauss_width, batch)
                
                #train_ds = IDSWDataSetLoader3(x_train, resolution,av_lum,model_name, config.gauss_range, config.gauss_width, device)# av_lum, res,pad,
                #train = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True) #, num_workers=2

                #                             x, res, av_lum, model_name, device
                # FOR increases DS size in training via augmentations (yaw augmentations) only create the DSL for test here, Train and Val in epoch loop
                # that will give different yaw augmentations each loop
                # if i also increase epochs, I get more unique tries for direction learning
                
                test_ds= IDSWDataSetLoader3(x_test, resolution,av_lum,model_name, config.gauss_range, config.gauss_width, device)
                test = DataLoader(test_ds, batch_size=config.batch_size, shuffle=True, drop_last=True) #, num_workers=2
               
                #val_ds= IDSWDataSetLoader3(x_val, resolution,av_lum,model_name, config.gauss_range, config.gauss_width, device)
                #val = DataLoader(val_ds, batch_size=config.batch_size, shuffle=True, drop_last=True) #, num_workers=2
                

                print(f"len training : {len(x_train)}     len val : {len(x_val)}    len test : {len(x_test)}")

                loss_fn = set_lossfn(loss)
                
                # set optimizer
                optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)

                wandb.watch(model, loss_fn, log='all', log_freq=2, idx = model_index)

                loop_run_name = f"{save_dict['Run']}_{resolution}_{config.learning_rate}_{scheduler_value}_{seed}_{loss}"


                
                # moving DL- only inputting x_train, x_val here
                model, save_dict = train_val_batch(model, train, x_val, loop_run_name,save_dict, config.learning_rate, loss_fn,epochs, config.batch_size, optimizer, scheduler_value, device)
                

                test_acc, test_predict_list, y_test = test_loop_batch(model,test, loss_fn, config.batch_size, device) 
                
                test_predict_numerical = [p.argmax().item() for p in test_predict_list]
                y_test_numerical = [y.argmax().item() for y in y_test]
                
                print(np.unique(test_predict_list))
                #print(np.unique(y_test))

                
                print(' \n train Acc: ', save_dict['t_accuracy_list'][-1])
                print(' \n val Acc: ', save_dict['v_accuracy_list'][-1])
                print(' \n test Acc: ', test_acc)
                
                save_dict.update({'test_acc': test_acc})
                save_dict.update({'test_predict': test_predict_list})
                save_dict.update({'test_labels': list(y_test)})
                #save_dict.update({'test_loss':test_loss})

                
                t_acc = save_dict['t_accuracy_list']
                tbase_acc, tMSE, tMAE, t_peakdist = getAcc_fromdict(t_acc)
                
                v_acc = save_dict['v_accuracy_list']
                vbase_acc, vMSE, vMAE, v_peakdist = getAcc_fromdict(v_acc)

                learning_curve(save_dict['t_loss_list'], save_dict['v_loss_list'], save_location=save_dict['save_location'],run_name=loop_run_name)
                
                accuracy_curve(tbase_acc, vbase_acc ,save_location=save_dict['save_location'],run_name="Basic"+loop_run_name)
                accuracy_curve(tMSE, vMSE ,save_location=save_dict['save_location'],run_name="MSE"+loop_run_name)
                accuracy_curve(tMAE, vMAE ,save_location=save_dict['save_location'],run_name="MAE"+loop_run_name)
                accuracy_curve(t_peakdist, v_peakdist ,save_location=save_dict['save_location'],run_name="PeakDist"+loop_run_name)
                
                #test_predict_list=[pred.cpu() for pred in test_predict_list]   ## pred is not currently a tensor. So not needed.
                                                                     ## If pred becomes a tensor - put it back in.

                print(f"test_predict_numerical :   {test_predict_numerical}")
                print(f"y_test_numerical :     {y_test_numerical}")
                plot_confusion(predictions= test_predict_numerical, actual= y_test_numerical, title = "Test Confusion matrix", run_name = loop_run_name,save_location =save_dict['save_location'])
                
                
                wandb.log({'test_predict': test_predict_list})
                wandb.log({'test_labels': list(y_test)})
                #saving
                diction = {}
                d = date.today()
                d=str(d)
                diction.update({'Date':d})
                diction.update({'gitHASH':str(gitHASH)})
                diction.update({'model_name': str(model_name)})
                diction.update({'loss_fn': str(loss)})
                diction.update({'lr': str(config.learning_rate)})
                #diction.update({'wd': str(wd_card)})
                #diction.update({'scheduler': str(scheduler_value)})
                diction.update({'seed': str(seed)})
                diction.update({'resolution': str(resolution)})
                #diction.update({'pad': int(pad)})
                diction.update({'lin_lay': int(lin_lay)})
                diction.update({'run time': (time.process_time() - run_start_time)})
                diction.update(save_dict)
                
                _save_location = save_dict['save_location']
                title = save_dict['Run']
                save2json(diction, loop_run_name, _save_location)
                save2csv(diction, title, _save_location)

                diction['model.state_dict'] = model.state_dict() #to('cpu').

                with open(f"{save_location}{loop_run_name}.pkl", 'wb+') as f:
                    pickle.dump(diction, f)
                
                #clear_output()
                
                print(f' \n END {model_name} {resolution} Run Time: ',time.process_time() - run_start_time)
                #!nvidia-smi
                torch.cuda.empty_cache()