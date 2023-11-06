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
import wandb
#


# loops



def loop(model, X, Y, epoch, loss_fn, device, col_dict, num_classes, pad_size =5, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
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
	pad = col_dict['padding']

	for idx, img in enumerate(X):
		#tense = tensoring(img).to(device)
		prepro = ImageProcessor(device)
		tense = prepro.colour_size_tense(img, colour, size, pad)


		prediction = model.forward(tense)
		label = label_oh_tf(Y[idx], device, num_classes)
		#if train:
		#	lr_ls.append(optimizer.param_groups[0]['lr'])
		loss = loss_fn(prediction, label)
		predict_list.append(prediction.argmax())

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



def test_loop(model, X, Y, loss_fn, device, col_dict,title, num_classes):
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
