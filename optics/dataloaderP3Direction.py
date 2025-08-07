# data set loader 3. adapted from IDSWDataSetLoader2 in functions.py
# imports
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import pprint
import csv
from datetime import date

class IDSWDataSetLoader3(dataset):
    def __intit__(self, x, res, av_lum, model_name, device):
        super(dataset, self).__init__()

        self.device = device
        self.imgpath = x
        #self.labels = y
        self.res = res
        self.pad = pad
        self.model_name = model_name
        self.av_lum = av_lum


        self.class_map = {
            "1" : 0,
            "2" : 1,
            "3" : 2,
            "4" : 3,
            "5" : 4, 
            "6" : 5,
            "7" : 6, 
            "8" : 7, 
            "9" : 8,
            "10" : 9,
            "11" : 10
        }

    def __len__(self):
        return len(self.imgpath)
            
    def tensoring(self, img):
        tense = torch.tensor(img, dtype=torch.float32)
        #tense = F.normalize(tense)
        tense = tense.permute(2, 0, 1)
        return tense

    def to_tensor(self, img):
        im_chan = img.shape[2]
        imgY, imgX = img.shape[0], img.shape[1]
        tensor = self.tensoring(img)
        tensor = tensor.reshape(im_chan, imgY, imgX)
        #print(' \n to tensor SELF.DEVICE: \n ', self.device)
        tensor = tensor.to(self.device)
        return tensor
        
    """def padding(self, img, pad_size):
        left_x = img[:,:pad_size,:] # h, w, c
        right_x = img[:,-pad_size:,:]
        y = img.shape[0]
        x = img.shape[1]+(pad_size*2)
        new_x = np.full((y, x, 3),255) # h w c
        new_x[:,:pad_size,:] = right_x
        new_x[:,pad_size:-pad_size,:] = img
        new_x[:,-pad_size:,:] = left_x
        return new_x"""
        
    def blank_padding(self, img, av_lum, final_size:tuple): 
        w = final_size[1]
        h = final_size[0]

        try:
            if img.shape[0] > h:
                img =cv2.resize(img, (img.shape[1],h), interpolation = cv2.INTER_NEAREST)
            if img.shape[1] > w:
                img =cv2.resize(img, (w, img.shape[0]), interpolation = cv2.INTER_NEAREST)
            #print("bp ",img.shape)
        except Exception as e:
            print(f"Error occurred: {e}")

        delta_w = w -img.shape[1]
        delta_h = h-img.shape[0]

        half_delta_h = int(np.floor(delta_h/2))
        half_delta_w = int(np.floor(delta_w/2))

        new_x = np.full((h,w,3), av_lum) 

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
        return new_x


     def colour_size_tense(self,image, vg =False):
        im = cv2.imread(image)
        LorR = np.random.randint(1,2)
        pixelOffset = random.randint(0,int(im.shape[1]/2))  # For random rotation shifting
        if LorR = 1:
            im = self.yaw(im, pixelOffset)
            label = self.gauss_label(180+pixelOffset)
        elif LorR = 2:
            im = self.yaw(im, -pixelOffset)
            label = self.gauss_label(180-pixelOffset) #label = self.gauss_label(self.labels[idx])
        
        im = cv2.resize(im, (self.res[0], self.res[1]))
        #if self.pad > 0: 
        #    im = self.padding(img=im, pad_size=self.pad)
        if vg:
            im = self.blank_padding(im, self.av_lum, (224,224)) 

        im = im/255 #norm
        
        im = self.to_tensor(im) 
        return im, label
         
    # label_oh_tf - BUT we want direction
    def gauss_label(self, north, gauss_range=45, gauss_width=3):
        """
        A function to produce a 360 degree label with a gaussian distribution of positive values (0-1) centered around 'north'.
        gauss range : number of degrees covered by gaussian distribution * 2 (left and right of peak).
        gauss_width : shape of guassian distribution (higher value is a wider curve, lower is sharper).
        """
        num_degrees = 360
        label = np.zeros(num_degrees)
        filtersize = (gauss_range*2)+1
        filtersize = int(np.floor(filtersize)) 
        depreciation = (gauss_range/gauss_width) # shape of curve
        gauss = gaussian(filtersize, depreciation)
        gauss /= gauss.max() # normalise 
    
        ## No bendy straights
        ## label[north-gauss_range:north+gauss_range+1] = gauss
    
        ### bendy straights version
        for i, value in enumerate(gauss):
            index = (north - gauss_range + i) % num_degrees
            label[index] = value
    
        return label

    def yaw(self, image, pixels):
        image = np.roll(image, pixels, axis=1)
        image[:,-1]= image[:,0]
        return image
        
    def __getitem__(self, idx, transform=False):
        # what object to return
        size= self.res
        pad = self.pad
        if self.model_name == 'vgg16' or self.model_name=='vgg':
            tense, label = self.colour_size_tense(self.img_path[idx], vg=True) 
        elif (self.model_name == '8c3l' and size == [57, 15]) or (self.model_name == '8c3l' and size == [29, 9]) or (self.model_name == '8c3l' and self.res == [15, 5]) or (self.model_name == '8c3l' and size ==[8, 3]):
            tense, label = self.colour_size_tense(self.img_path[idx], vg=True)
            
        elif (self.model_name == '7c3l' and size == [29, 9]) or (self.model_name == '7c3l' and self.res == [15, 5]) or (self.model_name == '7c3l' and size ==[8, 3]):
            tense, label = self.colour_size_tense(self.img_path[idx], vg=True)
        elif (self.model_name == '6c3l' and self.res == [15, 5]) or (self.model_name == '6c3l' and size ==[8, 3]): #and size == [29, 9]) or (self.model_name == '6c3l'
            tense, label = self.colour_size_tense(self.img_path[idx], vg=True)
        else:
            tense, label = self.colour_size_tense(self.img_path[idx])       

        # label
        return (tense, label)
        