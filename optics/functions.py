# 120923
# file for holding functions to keep notebooks clean

# Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
from sklearn.model_selection import train_test_split


#		GET DATA FUNCTIONS
def import_imagedata(file_path): # import image data from dir
	images = []
	labels = []

	#file_path = r'/its/home/nn268/optics/images/'

	for file in os.listdir(file_path):
		if file[0:4] == 'IDSW':
			j = file_path+file
			i=int(file[5:7]) -1
			i = str(i)
			labels.append(i)
			images.append(j)
	label_arr =np.array(labels)
	image_arr = np.array(images)
	return image_arr, label_arr

def get_data(file_path):
	x, y = import_imagedata(file_path)
	random_seed = random.seed()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, random_state=random_seed, shuffle=True)

	return x_train, y_train, x_val, y_val, x_test, y_test

# 		ONE HOT ENCODE LABEL DATA 
def label_oh_tf(lab, device, num_classes):	
	one_hot = np.zeros(num_classes)
	lab = int(lab)
	one_hot[lab] = 1
	label = torch.tensor(one_hot)
	label = label.to(torch.float32)
	label = label.to(device) #
	return label


# 	IMAGE DATA FUNCTIONS

def Unwrap(imgIn): #Amani unwrap fn

    def buildMap(Wd, Hd, R, Cx, Cy):
        ys=np.arange(0,int(Hd))
        xs=np.arange(0,int(Wd))

        rs=np.zeros((len(xs),len(ys)))
        rs=R*ys/Hd

        thetas=np.expand_dims(((xs-offset)/Wd)*2*np.pi,1)

        map_x=np.transpose(Cx+(rs)*np.sin(thetas)).astype(np.float32)
        map_y=np.transpose(Cy+(rs)*np.cos(thetas)).astype(np.float32)
        return map_x, map_y

    #UNWARP
    def Unwrap_(_img, xmap, ymap):
        output = cv2.remap(_img, xmap, ymap, cv2.INTER_LINEAR)
        return output


    img=cv2.resize(imgIn,None,fx=0.1,fy=0.1,interpolation=cv2.INTER_LINEAR)

    if img.shape[1] != img.shape[0]:
        cropBlock=int((int(img.shape[1])-int(img.shape[0]))/2)
        img=img[:,cropBlock:-cropBlock]

    #distance to the centre of the image
    offset=int(img.shape[0]/2)

    #IMAGE CENTER
    Cx = img.shape[0]/2
    Cy = img.shape[1]/2

    #RADIUS OUTER
    R =- Cx

    #DESTINATION IMAGE SIZE
    Wd = int(abs(2.0 * (R / 2)* np.pi))
    Hd = int(abs(R))

    #BUILD MAP
    xmap, ymap = buildMap(Wd, Hd, R, Cx, Cy)

    #UNWARP
    result = Unwrap_(img, xmap, ymap)

    return result
	

#	PREPROCESSING CLASS
# 	DEALS WITH: 	COLOUR	SCALE	 TENSOR
class  ImageProcessor():
	def __init__(self, device):
		self.device=device

	#  colour functions
	def two_channels(self, g, r):
		new_im = [[],[]]
		new_im[0] = g
		new_im[1] = r
		new_im = np.array(new_im)
		new_im = np.transpose(new_im, (1,2,0))
		return new_im
	# padding?
	def padding(self, img, pad_size):
		left_x = img[:,:pad_size,:] # h, w, c
		right_x = img[:,-pad_size:,:]
		y = img.shape[0]
		x = img.shape[1]+(pad_size*2)
		new_x = np.full((y, x, 3),255) # h w c
		new_x[:,:pad_size,:] = right_x
		new_x[:,pad_size:-pad_size,:] = img
		new_x[:,-pad_size:,:] = left_x
		return new_x

	# tenor functions
	def tensoring(self, img):
		tense = torch.tensor(img, dtype=torch.float32)
		tense = F.normalize(tense)
		tense = tense.permute(2, 0, 1)
		return tense

	def to_tensor(self, img):
		im_chan = img.shape[2]
		imgY, imgX = img.shape[0], img.shape[1]
		tensor = self.tensoring(img)
		tensor = tensor.reshape(1, im_chan, imgY, imgX)
		tensor = tensor.to(self.device)
		return tensor

	#useful functions
	def colour_size_tense(self, img_path, col, size, pad:int, unwrap=False):
		if isinstance(img_path, str):
			im = cv2.imread(img_path)
			#print(im.shape, '1')
			#print(im)
		else:
			im= img_path
			
		if unwrap:
			if size[0] != size[1]:
				im = Unwrap(im)
				#print(im.shape, '2')
		#print(im, '2')
		if im.shape[2]==1:
			#im= cv2.resize(im, (size[0], size[1]))
			im= self.to_tensor(im)
			return(im)
		r = im[:,:,2]
		g = im[:,:,1]
		b = im[:,:,0]

		if col == 'nored':
			im = self.two_channels(b, g)
		elif col == 'noblue':
			im = self.two_channels(g, r)
		elif col == 'nogreen':
			im = self.two_channels(b, r)
		elif col == 'grey':
			im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		elif col =='colour' or col == 'color':
			pass
		
		if unwrap:
			im = cv2.resize(im, (size[0], size[1]))
			#print(im.shape, '3')

		if pad > 0:
			im = self.padding(img=im, pad_size=pad)
			#print(im.shape, '4')
		#print(im.shape, '5')
		im = self.to_tensor(im)
		#print(type(im))
		return im

	def view(self, img, scale:int):
		if type(img) == torch.Tensor:
			img = img.squeeze()
			img = img.permute(1,2,0)
			img=np.array(img.cpu())*scale
			plt.imshow(img)
			plt.axis(False)
			plt.show()
			return img






def add_padding(img, pad_size): 
    # add padding to unwrapped tensor image
    img = img.squeeze()
    # select padding from sides of image
    left_x = img[:,:,:pad_size]
    right_x = img[:,:,-pad_size:]
    # get sizes for new image
    y = img.shape[1]
    x = img.shape[2]+(pad_size*2)
    # create empty array for new image size
    new_x = np.zeros((3, y, x))
    # fill empty array
    new_x[:,:,:pad_size] = right_x
    new_x[:,:,pad_size:-pad_size] = img
    new_x[:,:,-pad_size:] = left_x
    # convert to tensor
    new_x = torch.tensor(new_x, dtype=torch.float32)
    new_x = torch.unsqueeze(new_x, 0)
    return new_x


def yaw(image, pixels):
        image = np.roll(image, pixels, axis=1)
        image[:,-1]= image[:,0]
        return image

# Helpful printing functions. Could probably be deleated
def print_run_header(learning_rate, optim, loss_fn):
	print('\n')
	print('LR: ', learning_rate)
	print('optimiser ', optim)
	print('loss fn: ', loss_fn)

def print_run_type(run_type: str):
	print('                  ----------------------')
	print(f' \n                  {run_type}... \n')
	print('                  ----------------------')

def check_best_accuracy(v_accuracy_list, best_valaccuracy):
	if v_accuracy_list[-1] > best_valaccuracy:
		best_valaccuracy = v_accuracy_list[-1]
		best_optim = optimizer
		best_lossfn = loss_fn
		best_lr = learning_rate
		best_epoch = epoch
	return best_valaccuracy, best_optim, best_lossfn, best_lr, best_epoch

def print_top_results(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch):
	print('Top results from hyperparameter sweep:')
	print()
	print(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch)





