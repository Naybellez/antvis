import cv2
from PIL import Image

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as maths

import os
import random

from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import torch
import torch.nn as nn
from torch.nn import functional
#from torchsummary import summary
#import torchvision.transforms as transforms

from tqdm import tqdm

from functions import ImageProcessor,label_oh_tf
#import wandb
#


# loops

def loop(model, X, Y, loss_fn, device, size, pad, num_classes, model_name, av_lum,colour= 'colour', optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
    model = model
    total_samples = len(X)
    if train:
        model.train()
        #lr_ls = []
    else:
        model.eval()

    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0

    #print(model_name)

    for idx, img in enumerate(X):
        #tense = tensoring(img).to(device)
        prepro = ImageProcessor(device)
        #print('loop size: ',size, type(size))
        if model_name == 'vgg16':
            #if col_dict['size'][0] >= 224 or col_dict['size'][1] >= 224: 
            #print('vgg registered')
            tense = prepro.colour_size_tense(img, colour, size,av_lum, pad, vg=True) #[29, 9], 15, 5, [8,3]
        elif (model_name == '7c3l' and size == [29, 9]) or (model_name == '7c3l' and size == [15, 5]) or (model_name == '7c3l' and size ==[8, 3]):
            #print('7c and small size registered')
            tense = prepro.colour_size_tense(img, colour, size, pad, vg=True)
        else:
            #print('coloursizetense as norm registered')
            tense = prepro.colour_size_tense(img, colour, size,av_lum, pad)
        #print(tense.shape)

        prediction = model.forward(tense)
        label = label_oh_tf(Y[idx], num_classes).to(device)
        #if train:
        #	lr_ls.append(optimizer.param_groups[0]['lr'])
        loss = loss_fn(prediction, label)
        predict_list.append(prediction.argmax())
        #print('loop loss: ',loss.item())
        if prediction.argmax() == label.argmax():
            num_correct +=1
            #if train:
            #	print(f'\n ########################### HIT ###########################  -- {idx} / {total_samples} \n')
        total_count+=1
        current_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
    #print(num_correct/len(X))
    if train:
        return current_loss, predict_list, num_correct, model, optimizer #, lr_ls
    else:
        return current_loss, predict_list, num_correct



def test_loop(model, X, Y, loss_fn, device, title, num_classes):
	model = model.eval()
	predict_list = []
	total_count =0
	num_correct = 0
	correct = 0
	colour = col_dict['colour']
	size = col_dict['size']

	with torch.no_grad():
		for idx, img in enumerate(X):
			prepro = ImageProcessor(device)
			tense = prepro.colour_size_tense(img, colour, size)
			prediction = model.forward(tense)
			label = label_oh_tf(Y[idx], device, num_classes)

			if prediction.argmax()==label.argmax():
				num_correct +=1
			total_count +=1
			correct +=(prediction.argmax()==label.argmax()).sum().item()

		acc = num_correct/total_count
		accuracy = 100*(acc)

		X = list(X)
		log_test_score(acc, accuracy, X)


def loop_og(model, X, Y, loss_fn, device, col_dict, num_classes, model_name, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
    model = model
    total_samples = len(X)
    if train:
        model.train()
        #lr_ls = []
    else:
        model.eval()

    predict_list = []
    total_count = 0
    num_correct = 0
    current_loss = 0
    colour = col_dict['colour']
    size = col_dict['size']
    pad = col_dict['pad']
    av_lum = col_dict['av_lum']

    for idx, img in enumerate(X):
        #tense = tensoring(img).to(device)
        prepro = ImageProcessor(device)
        tense = prepro.colour_size_tense(img, colour, size, av_lum, pad)

        #print(tense.shape)
        prediction = model.forward(tense)
        label = label_oh_tf(Y[idx], num_classes).to(device)
        #if train:
        #	lr_ls.append(optimizer.param_groups[0]['lr'])
        loss = loss_fn(prediction, label)
        predict_list.append(prediction.argmax())

        if prediction.argmax() == label.argmax():
            num_correct +=1
            #if train:
            #	print(f'\n ########################### HIT ###########################  -- {idx} / {total_samples} \n')
        total_count+=1
        
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
        current_loss += loss.item()
    #print(num_correct/len(X))
    if train:
        return current_loss, predict_list, num_correct, model, optimizer #, lr_ls
    else:
        return current_loss, predict_list, num_correct



def test_loop_og(model, X, Y, loss_fn, device, col_dict,title, num_classes):
	model = model.eval()
	predict_list = []
	total_count =0
	num_correct = 0
	correct = 0
	colour = col_dict['colour']
	size = col_dict['size']

	with torch.no_grad():
		for idx, img in enumerate(X):
			prepro = ImageProcessor(device)
			tense = prepro.colour_size_tense(img, colour, size)
			prediction = model.forward(tense)
			label = label_oh_tf(Y[idx], device, num_classes)

			if prediction.argmax()==label.argmax():
				num_correct +=1
			total_count +=1
			correct +=(prediction.argmax()==label.argmax()).sum().item()

		acc = num_correct/total_count
		accuracy = 100*(acc)

		X = list(X)
		log_test_score(acc, accuracy, X)
