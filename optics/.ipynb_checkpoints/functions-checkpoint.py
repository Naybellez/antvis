# 120923
# file for holding functions to keep notebooks clean

# Imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch.nn import functional as F
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pprint
import csv
import json
from datetime import date
import collections
pp = pprint.PrettyPrinter(indent=4)


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

def get_data(file_path, seed):
	x, y = import_imagedata(file_path)
	random_seed = random.seed(seed)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, random_state=random_seed, shuffle=True)

	#train_loader = DataLoader(list(zip(x_train, y_train)), shuffle =True, batch_size=16) # machinelearningmastery.com
	#val_loader = DataLoader(list(zip(x_val, y_val)), shuffle =True, batch_size=16)
	#test_loader = DataLoader(list(zip(x_test, y_test)), shuffle = True, batch_size=16)
	return x_train, y_train, x_val, y_val, x_test, y_test
	#return train_loader, val_loader, test_loader

# 		ONE HOT ENCODE LABEL DATA 
def label_oh_tf(lab, num_classes):	#device,
	one_hot = np.zeros(num_classes)
	lab = int(lab)
	one_hot[lab] = 1
	label = torch.tensor(one_hot)
	label = label.to(torch.float32)
	#label = label.to(device) #
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

from numpy.linalg import norm
#   PREPROCESSING CLASS
#   DEALS WITH: 	COLOUR	SCALE	 TENSOR
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

    

    def luminance(self, img):
        r,g,b = self.split_channels(img)
        lum = (0.114*b)+(0.587*g)+(0.299*r)
        mean_lum = np.mean(lum)
        return mean_lum
    
    def blank_padding(self, img, final_size:list): 

        w = final_size[1]
        h = final_size[0]
        try:
            if img.shape[0] > h:
                img =cv2.resize(img, (img.shape[1],h), interpolation = cv2.INTER_NEAREST)

            if img.shape[1] > w:
                img =cv2.resize(img, (w, img.shape[0]), interpolation = cv2.INTER_NEAREST)
  
        except Exception as e:
            print(f"Error occurred: {e}")

        

        delta_w = w -img.shape[1]
        delta_h = h-img.shape[0]

        half_delta_h = int(np.floor(delta_h/2))
        half_delta_w = int(np.floor(delta_w/2))

        avg_lum = int(self.luminance(img)) 
        new_x = np.full((h,w,3), avg_lum) 
        if img.shape[1]%2 ==0: 
            if img.shape[0]%2 == 0: 
                if half_delta_w == 0:
                    if half_delta_h ==0:
                        new_x[:,:,:] = img # h=72 w=224
                    else:
                        new_x[half_delta_h:-half_delta_h,:,:] = img
                else:
                    new_x[half_delta_h:-half_delta_h,half_delta_w:-half_delta_w,:] = img
            else:
                new_x[half_delta_h:-(half_delta_h+1),half_delta_w:-half_delta_w,:] = img
        else:
            if img.shape[0]%2 == 0:
                new_x[half_delta_h:-half_delta_h,half_delta_w:-(half_delta_w+1),:] = img #*#*#
            else:
                new_x[half_delta_h:-(half_delta_h+1),half_delta_w:-(half_delta_w+1),:] = img
        print(new_x.shape)
        return new_x


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
        #tensor = tensor.reshape(1, im_chan, imgY, imgX)
        tensor = tensor.reshape(im_chan, imgY, imgX)
        #print('to tensor SELF.DEVICE: \n ',self.device)
        tensor = tensor.to(self.device)
        return tensor
    def split_channels(self, im):
            r = im[:,:,2]
            g = im[:,:,1]
            b = im[:,:,0]
            return r,g,b
    def im_channels(self,im,col):
        r,g,b = self.split_channels(im)
        if col.lower() == 'nored':
            im = self.two_channels(b, g)
        elif col.lower() == 'noblue':
            im = self.two_channels(g, r)
        elif col.lower() == 'nogreen':
            im = self.two_channels(b, r)
        elif col.lower() == 'grey':
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        elif col.lower() =='colour' or col == 'color':
            pass
        return im

    #useful functions
    def colour_size_tense(self, img_path, col:str, size, pad:int,vg =False, unwrap=False):

        if isinstance(img_path, str):
            im = cv2.imread(img_path)
        else:
            im= img_path


        if unwrap: 
            if size[0] != size[1]: 
                im = Unwrap(im)
        #print(im.shape)
        if im.shape[2]==1: 
            im= self.to_tensor(im)
            return(im)


        im = self.im_channels(im,col)

        im = cv2.resize(im, (size[0], size[1])) # resize the image

        if pad > 0: # if padding has been specified...
            im = self.padding(img=im, pad_size=pad)
        if vg:
            im = self.blank_padding(im, (224,224)) 
        #print(im.shape)
        im = self.to_tensor(im) 
        #print(im.shape)
        return im

    def view(self, img, scale:int):
        if type(img) == torch.Tensor:
            img = img.squeeze()
            img = img.permute(1,2,0)
            img=np.array(img.cpu())*scale

        elif type(img) == np.ndarray:
            img = img*scale
        elif type(img) == str:
            cv2.imread(img)
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

"""
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

	#train_loader = DataLoader(list(zip(x_train, y_train)), shuffle =True, batch_size=16) # machinelearningmastery.com
	#val_loader = DataLoader(list(zip(x_val, y_val)), shuffle =True, batch_size=16)
	#test_loader = DataLoader(list(zip(x_test, y_test)), shuffle = True, batch_size=16)
	return x_train, y_train, x_val, y_val, x_test, y_test
	#return train_loader, val_loader, test_loader
"""

class IDSWDataSetLoader(Dataset):
    def __init__(self, x, y, col_dict, device): # transform =True
        super(Dataset, self).__init__()

        self.device = device
        self.col_dict = col_dict

        self.img_path = x
        self.labels = y

        self.class_map = {"1":0,"2": 1,
                            "3":2, "4":3,
                            "5":4, "6": 5,
                            "7":6, "8":7,
                            "9":8, "10": 9,
                            "11":10}


    def __len__(self):
        # length of dataset
        return len(self.img_path)
    
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
        tensor = tensor.reshape(im_chan, imgY, imgX)
        print(' \n to tensor SELF.DEVICE: \n ', self.device)
        tensor = tensor.to(self.device)
        return tensor

    def __getitem__(self, idx, transform=False):
        # what object to return
        size= self.col_dict['size']
        pad = self.col_dict['padding']
        
        img = cv2.imread(self.img_path[idx])
        self.transform = transform

        im_chan = img.shape[2]
        if size:
            img = cv2.resize(img, (size[0], size[1]))
            h = size[1]
            w = size[0]
        else:
            h = img[0]
            w = img[1]

        img = img/255 #norm

        tense = self.to_tensor(img)

        label = label_oh_tf(self.labels[idx], 11)
        return tense, label

    
def save2csv(nested_dict, file_name, save_location:str):
    columns = list(nested_dict.keys())
    path = os.path.join(save_location, file_name +".csv")
    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            # using dictwriter
            # using writeheader function
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(nested_dict)
            f.close()
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError:
              print("could not convert to string")
    except:
              print("unexpected error: ", sys.exc_info()[0])
        

def save2json(nested_dict, file_name, save_location:str):
    json_obj = json.dumps(nested_dict, indent=4)
    print(json_obj)
    path = os.path.join(save_location, file_name+".json")
    #print(path)
    with open(path, 'w') as f:
        f.write(json_obj)
        

def read_in_json(file_path, file_name):
    path = os.path.join(file_path, 'file_name')
    try:
        with open(path, 'r') as f:
            #obj = f.read()
            dj = json.load(f, object_pairs_hook= collections.OrderedDict) #obj, 
            print(dj)
    except Exception as e:
        print("Error decoding Json")
        print(e)
    
#prepro = ImageProcessor(device)
#self.x_train = prepro.colour_size_tense(x_train, col_dict['colour'], col_dict['size'], col_dict['pad'])
#self.x_val = prepro.colour_size_tense(x_val, col_dict['colour'], col_dict['size'], col_dict['pad'])
#self.x_test = prepro.colour_size_tense(x_test , col_dict['colour'], col_dict['size'], col_dict['pad'])

#self.y = label_oh_tf(11)
"""train_loader = DataLoader(
	list(zip(self.x_train, self.y_train)),
	shuffle=True,
	batch_size=16
	)
val_loader = DataLoader(
	list(zip(self.x_val, self.y_val)),
	shuffle=True,
	batch_size=16
	)
test_loader = DataLoader(
	list(zip(self.x_test, self.y_test)),
	shuffle=True,
	batch_size=16
	)
return train_loader, test_loader, val_loader"""



"""		def _import_imagedata(self, file_path): 
# import image data from dir
images = []
labels = []

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

def get_data(self, file_path):
# split data into sets
x_arr, y_arr = self._import_imagedata(file_path)
self.n_samples = len(x_arr) # get len of whole dataset

random_seed = random.seed(3)

x_train, x_test, y_train, y_test = train_test_split(x_arr, y_arr, test_size=0.3, random_state=random_seed)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size =0.1, random_state=random_seed, shuffle=True)

return x_train, y_train, x_val, y_val, x_test, y_test
"""