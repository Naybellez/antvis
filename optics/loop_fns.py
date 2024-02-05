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
import subprocess
from functions import ImageProcessor,label_oh_tf
import wandb
#


# loops

def print_gpu_mem():
	proc = subprocess.run(['nvidia-smi', '-i', '0'], capture_output=True, text=True)
	output = proc.stdout
	p2 = subprocess.run(['grep', '/usr'], input=output, capture_output=True, text=True)
	filtered_output = p2.stdout
	print(filtered_output)


def loop(model, x, y, epoch, loss_fn, device, col_dict, num_classes, pad_size =5, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
	#model = model
	if train:
		model.train()
	else:
		model.eval()

	predict_list = []
	label_list = []
	total_count = 0
	num_correct = 0
	current_loss = 0
	colour = col_dict['colour']
	size = col_dict['size']
	pad = col_dict['padding']

	prepro = ImageProcessor(device)

	for idx, img in enumerate(x):

		tense = prepro.colour_size_tense(img, colour, size, pad)

		prediction = model.forward(tense) # tense
		label = label_oh_tf(y[idx], num_classes).to(device)
		loss = loss_fn(prediction, label)
		

		if prediction.argmax() == label.argmax():
			num_correct +=1
		
		total_count+= 1 # **
		
		
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			if scheduler:
				scheduler.step()

		# * current loss +=
		current_loss += loss.item()  # *

		#loss = loss.to('cpu')
		#current_loss += loss.item()

		predict_list.append(prediction.argmax().to('cpu'))
		label_list.append(label.argmax().to('cpu'))#(label.to('cpu'))

	if train:
		return current_loss, predict_list, num_correct, label_list, model, optimizer #, lr_ls
	else:
		return current_loss, predict_list, num_correct, label_list



"""def loop(model, x, y, epoch, loss_fn, device, col_dict, num_classes, pad_size =5, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
	#model = model
	#print('in loop, col_dict is a: ',type(col_dict), col_dict)
	#total_samples = 0 #len(X) #?

	
	if train:
		model.train()
		#lr_ls = []
	else:
		model.eval()

	predict_list = []
	label_list = []
	total_count = 0
	num_correct = 0
	current_loss = 0
	colour = col_dict['colour']
	size = col_dict['size']
	pad = col_dict['padding']

	prepro = ImageProcessor(device)

	#x_batch= x_batch.to(device)
	for idx, img in enumerate(x):
		#print(img)
		
		#tense = tensoring(img).to(device)
		#if idx == 0:
		#	display=True
		#else:
		#	display = False

		tense = prepro.colour_size_tense(img, colour, size, pad)
		#print(tense.shape, type(tense))
		#prepro.view(tense, 5)
		#print('label raw: ',y[idx])
		#if display:
		#	print('train?', train, idx)
		#	print('Avatar Whan')
		#	print_gpu_mem()

		prediction = model.forward(tense) # tense
		#print('y', y)
		label = label_oh_tf(y[idx], num_classes).to(device)
		#print('img number: ',img[59:61])
		#print('label: ', y[idx])
		
		#if train:
		#	lr_ls.append(optimizer.param_groups[0]['lr'])
		loss = loss_fn(prediction, label)
		#if display:
		#	print('Avatar Yangchen')
		#	print_gpu_mem()
		


		#if prediction.argmax() == label.argmax():
		#	num_correct +=1
		if prediction.argmax() == label.argmax():
			num_correct +=1
			#if train:
			#	print(f'\n ########################### HIT ###########################  -- {idx} / {total_samples} \n')
		#total_count+=1
		#if display:
		#	print('Avatar Kuruk')
		#	print_gpu_mem()
		
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#if display:
			#	print('Avatar Kioshi')
			#	print_gpu_mem()
			if scheduler:
				scheduler.step()
		#if display:		
		#	print('Avatar Roku')
		#	print_gpu_mem()

		loss = loss.to('cpu')
		#print('loss item: ', loss.item())
		current_loss += loss.item()
		#if display:
		#	print('Avatar Aang')
		#	print_gpu_mem()
		#loss_list.append(loss)
		predict_list.append(prediction.argmax().to('cpu'))
		label_list.append(label.argmax().to('cpu'))#(label.to('cpu'))
		#label_list.append(y_batch.to('cpu'))#(label.to('cpu'))
		#if display:
		#	print('Avatar Korra')
		#	print_gpu_mem()

	#print(num_correct/len(X))
	#print('current loss: ',current_loss)
	if train:
		return current_loss, predict_list, num_correct, label_list, model, optimizer #, lr_ls
	else:
		return current_loss, predict_list, num_correct, label_list"""

# editing loop to encorperate dataloader
def batch_loop(model, loader, epoch, loss_fn, device, col_dict, num_classes, pad_size =5, optimizer =None, scheduler= None, train =True):	# Train and Val loops. Default is train
	#model = model
	#print('in loop, col_dict is a: ',type(col_dict), col_dict)
	#total_samples = 0 #len(X) #?

	
	if train:
		model.train()
		#lr_ls = []
	else:
		model.eval()

	predict_list = []
	label_list = []
	total_count = 0
	num_correct = 0
	current_loss = 0
	colour = col_dict['colour']
	size = col_dict['size']
	pad = col_dict['padding']

	#x_batch= x_batch.to(device)
	#for idx, img in enumerate(X):
	for x_batch, y_batch in loader:
		x_batch= x_batch.to(device)
		y_batch = y_batch.to(device)
		y_batch =y_batch.argmax()
		#tense = tensoring(img).to(device)
		#if idx == 0:
		#	display=True
		#else:
		#	display = False

		#tense = prepro.colour_size_tense(img, colour, size, pad)
		#if display:
		#	print('train?', train, idx)
		#	print('Avatar Whan')
		#	print_gpu_mem()

		prediction = model.forward(x_batch) # tense
		#label = label_oh_tf(y_batch, num_classes).to(device)
		
		#if train:
		#	lr_ls.append(optimizer.param_groups[0]['lr'])
		loss = loss_fn(prediction, y_batch)
		#if display:
		#	print('Avatar Yangchen')
		#	print_gpu_mem()
		


		#if prediction.argmax() == label.argmax():
		#	num_correct +=1
		if prediction.argmax() == y_batch.argmax():
			num_correct +=1
			#if train:
			#	print(f'\n ########################### HIT ###########################  -- {idx} / {total_samples} \n')
		#total_count+=1
		#if display:
		#	print('Avatar Kuruk')
		#	print_gpu_mem()
		
		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			#if display:
			#	print('Avatar Kioshi')
			#	print_gpu_mem()
			if scheduler:
				scheduler.step()
		#if display:		
		#	print('Avatar Roku')
		#	print_gpu_mem()

		#loss = loss.to('cpu')
		current_loss += loss.item()
		#if display:
		#	print('Avatar Aang')
		#	print_gpu_mem()
		#loss_list.append(loss)
		predict_list.append(prediction.argmax().to('cpu'))
		label_list.append(y_batch.to('cpu'))#(label.to('cpu'))
		#if display:
		#	print('Avatar Korra')
		#	print_gpu_mem()

	#print(num_correct/len(X))
	print(current_loss)
	if train:
		return current_loss, predict_list, num_correct, label_list, model, optimizer #, lr_ls
	else:
		return current_loss, predict_list, num_correct, label_list



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
			tense = prepro.colour_size_tense(img, colour, size, pad=5)
			prediction = model.forward(tense)
			label = label_oh_tf(Y[idx], num_classes)

			if prediction.argmax()==label.argmax():
				num_correct +=1
			total_count +=1
			correct +=(prediction.argmax()==label.argmax()).sum().item()

		acc = num_correct/total_count
		accuracy = 100*(acc)

		#X = list(X)
		#log_test_score(acc, accuracy, X)
		print('TEST ACCURACY: ',accuracy)
	return predict_list, Y, accuracy

def test_loop_batch(model, loader, loss_fn, device, col_dict,title, num_classes):
	model = model.eval()
	predict_list = []
	total_count =0
	num_correct = 0
	correct = 0
	colour = col_dict['colour']
	size = col_dict['size']

	with torch.no_grad():
		for x_batch, y_batch in loader:
			#prepro = ImageProcessor(device)
			#tense = prepro.colour_size_tense(img, colour, size, pad=5)
			prediction = model.forward(tense)
			#label = label_oh_tf(Y[idx], num_classes)

			if prediction.argmax()==y_batch.argmax():
				num_correct +=1
			total_count +=1
			correct +=(prediction.argmax()==y_batch.argmax()).sum().item()

		acc = num_correct/total_count
		accuracy = 100*(acc)

		#X = list(X)
		#log_test_score(acc, accuracy, X)
		print('TEST ACCURACY: ',accuracy)
	return predict_list, y_batch, accuracy
