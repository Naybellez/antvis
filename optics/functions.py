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
from datetime import date

pp = pprint.PrettyPrinter(indent=4)


#	#	GET DATA FUNCTIONS  #	#	GET DATA FUNCTIONS  #	#	GET DATA FUNCTIONS  #	#	GET DATA FUNCTIONS
def import_imagedata(file_path): 
    # import image data from dir
    images = []
    labels = []
    print(file_path)

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

def get_data(random_seed, file_path):
    #print(file_path)
    img_len = len(os.listdir(file_path))
    x, y = import_imagedata(file_path)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, train_size=0.7,
                                     random_state=random_seed, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.3, train_size=0.7,
                                     random_state=random_seed, shuffle=True)
    return x_train, y_train, x_val, y_val, x_test, y_test


# 	#	ONE HOT ENCODE LABEL DATA   # 	#	ONE HOT ENCODE LABEL DATA   # 	#	ONE HOT ENCODE LABEL DATA   # 	
def label_oh_tf(lab, num_classes):	#device,
	one_hot = np.zeros(num_classes)
	lab = int(lab)
	one_hot[lab] = 1
	label = torch.tensor(one_hot)
	label = label.to(torch.float32)
	return label


# 	IMAGE DATA FUNCTIONS # 	IMAGE DATA FUNCTIONS # 	IMAGE DATA FUNCTIONS # 	IMAGE DATA FUNCTIONS

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


    def new_luminance(self, dataset):
        data_len = len(dataset)
        r= []
        b =[]
        g = []
        for idx, img_path in enumerate(dataset):
            d = cv2.imread(img_path)
            if d is None:
                #print('boo')
                print('Bad path:  ',img_path)
                continue
            r_,g_,b_ = self.split_channels(d)
            r.append(r_)
            g.append(g_)
            b.append(b_)
        av_r = sum(r)/len(r)
        av_b = sum(b)/ len(b)
        av_g = sum(g) / len(g)
        lum = (0.114*av_b)+(0.587*av_g)+(0.299*av_r)
        mean_lum = np.mean(lum)
        return mean_lum
    
    def blank_padding(self, img, av_lum, final_size:list): 

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

        #avg_lum = int(self.luminance(img)) 
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
        #print(new_x.shape)
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
        img = img/255
        tense = torch.tensor(img, dtype=torch.float32)
        #tense = F.normalize(tense)
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
    def colour_size_tense(self, img_path, col:str, size, av_lum,  pad:int,vg =False, unwrap=False):

        if isinstance(img_path, str):
            im = cv2.imread(img_path)
            #plt.imshow(im)
            #plt.show()
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
            #print('vg in place')
            im = self.blank_padding(im, av_lum, (224,224)) 
        #print(im.shape)
        #plt.imshow(im)
        #plt.show()
        im = self.to_tensor(im) 
        #print(im.shape)
        return im

    def trans_to_img(self, img, scale):
        if isinstance(img, torch.Tensor):  #type(img) == torch.Tensor:
            img = img.squeeze()
            img = img.permute(1,2,0)
            img=np.array(img.cpu())*scale

        elif isinstance(img, np.ndarray): # type(img) == np.ndarray:
            img = img*scale
        elif isinstance(img, str): # type(img) == str:
            #print("here 1")
            img = cv2.imread(img)
            #print("here 2")
        return img

    def view(self, img, scale:int, loop_run_name:str, save_dict:dict,  epoch:int, where:str):
        img = self.trans_to_img(img, scale)
        if save_dict != None:
            res = cv2.normalize(img, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imwrite(f"{save_dict['save_location']}_randImg{loop_run_name}_{epoch}_{where}.png", res) #*255
            #plt.imsave(res)
            #plt.savefig
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

class IDSWDataSetLoader2(Dataset):
    def __init__(self, x, y, res,pad,av_lum, model_name, device): # transform =True
        super(Dataset, self).__init__()

        self.device = device
        #self.col_dict = col_dict

        self.img_path = x
        self.labels = y
        self.res = res
        self.pad = pad
        self.model_name = model_name
        self.av_lum =av_lum

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

    def label_oh_tf(self, lab):	#device,
        one_hot = np.zeros(11)
        lab = int(lab)
        one_hot[lab] = 1
        label = torch.tensor(one_hot)
        label = label.to(torch.float32)
        label= label.to(self.device)
        return label
        
    def colour_size_tense(self,image, vg =False):
        im = cv2.imread(image)
        im = cv2.resize(im, (self.res[0], self.res[1]))
        if self.pad > 0: 
            im = self.padding(img=im, pad_size=self.pad)
        if vg:
            im = self.blank_padding(im, self.av_lum, (224,224)) 

        im = im/255 #norm
        im = self.to_tensor(im) 
        return im
        
    def __getitem__(self, idx, transform=False):
        # what object to return
        size= self.res
        pad = self.pad
        if self.model_name == 'vgg16' or self.model_name=='vgg':
            tense = self.colour_size_tense(self.img_path[idx], vg=True) 
        elif (self.model_name == '7c3l' and size == [29, 9]) or (self.model_name == '7c3l' and self.res == [15, 5]) or (self.model_name == '7c3l' and size ==[8, 3]):
            tense = self.colour_size_tense(self.img_path[idx], vg=True)
        elif (self.model_name == '6c3l' and self.res == [15, 5]) or (self.model_name == '6c3l' and size ==[8, 3]): #and size == [29, 9]) or (self.model_name == '6c3l'
            tense = self.colour_size_tense(self.img_path[idx], vg=True)
        else:
            tense = self.colour_size_tense(self.img_path[idx])        
        label = self.label_oh_tf(self.labels[idx])
        return tense, label

class IDSWDataSetLoader7(Dataset):
    def __init__(self, img_path, labels, av_lum, transform=None, res = (452, 144), vgg=False): 
        super(Dataset, self).__init__()
        self.img_path = img_path
        self.labels = labels
        self.av_lum = av_lum
        self.transform= transform
        self.res = res
        self.vgg = vgg
        
    def __len__(self):
        # length of dataset
        return len(self.labels)
        pass

    def get_padding(self, img,  final_size=(224, 224)): 
        import cv2
        import numpy as np
        output_height = final_size[0]
        output_width = final_size[1]
    
        try:
            if img.shape[0] > output_height:
                img = cv2.resize(img, (output_height, img.shape[1]), interpolation = cv2.INTER_NEAREST) # this might be the issue
            if img.shape[1] > output_width:
                img = cv2.resize(img, (img.shape[0], output_width), interpolation = cv2.INTER_NEAREST)
    
        except Exception as e:
            print(f"Image Resizing Error Occurred: {e}")
    
        image_height = img.shape[0]
        image_width = img.shape[1]
    
        delta_width = output_width - image_width
        delta_height = output_height - image_height
    
        half_delta_height = int(np.floor(delta_height/2))
        half_delta_width = int(np.floor(delta_width/2))

        if isinstance(delta_height/2, int):
            half_delta_height1, half_delta_height2 = half_delta_height, half_delta_height
        else:
            half_delta_width1, half_delta_width2 = half_delta_width, (half_delta_width+1)
        if isinstance(delta_width/2, int):
            half_delta_height1, half_delta_height2 = half_delta_height, half_delta_height
        else:
            half_delta_height1, half_delta_height2 = half_delta_height, (half_delta_height+1)
        
        return half_delta_height1, half_delta_height2, half_delta_width1, half_delta_width2
    
    def __getitem__(self, idx):
        import cv2
        from PIL import Image
        from torchvision import transforms
       # read in image with cv2 so that padding calculations can occur (array)
        img = cv2.imread(self.img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        # resize
        new_lum = int(round(self.av_lum*255))
        yaw_padded_img, yaw_padded_label = self.Yaw_padding(new_lum)({"image": img, "label": label})
        img = cv2.resize(yaw_padded_img, (self.res[0], self.res[1]))


        transform = transforms.Compose([
                transforms.PILToTensor()
            ])
            
        if self.vgg==True:
            
            
            h_delta_height1, h_delta_height2, h_delta_width1, h_delta_width2 = self.get_padding(img)
            print("here are the padding values: ",h_delta_width1, h_delta_height1, h_delta_width2, h_delta_height2)
            pil_im = Image.fromarray(img, mode="RGB")
            # create new image with set size coloured with average luminance
            pil_res2 = Image.new(pil_im.mode, (224, 224), (new_lum, new_lum, new_lum))  #(int(av_lum), int(av_lum), int(av_lum)
            # paste the resized image (from array) onto the grey padded background image
            pil_res2.paste(pil_im, (h_delta_width1, h_delta_height1, -h_delta_width2, -h_delta_height2))
            ## at this point, the image looks good
            # convert the PIL image to a tensor
            tans_img = transform(pil_res2)
        else:
            tans_img = Image.fromarray(img, mode="RGB")
            tans_img = transform(tans_img)
        tans_img = tans_img/255
        label = self.Label_oh_tf()({"image": img, "label": label})
        return tans_img, label

    class PrintShape(object):
        def __call__(self, sample):
            image, lab = sample['image'], sample['label']
            return image.shape
    
    class Permute_im(object): 
        def __call__(self, sample):
            
            image, lab = sample['image'], sample['label']
            image = F.normalize(image)
    
            return image, lab #
    
    class Label_oh_tf(object):	#device,
        def __call__(self, sample):
            import numpy as np
            import torch
            image, lab = sample['image'], sample['label'] 
            one_hot = np.zeros(11)
    
            lab = int(lab)
            one_hot[lab] = 1
            label = torch.tensor(one_hot)
            label = label.to(torch.float32)
            return image, label
    
    
    class Yaw_padding(object):
        import numpy as np
        def __init__(self, av_lum):
            self.av_lum=av_lum
        def __call__(self, sample, pad_size=3):
            img, lab = sample['image'], sample['label']
            left_x = img[:,:pad_size,:] # h, w, c
            right_x = img[:,-pad_size:,:]
            y = img.shape[0]
            x = img.shape[1]+(pad_size*2)
            new_x = np.full((y, x, 3),self.av_lum, dtype=np.uint8) # h w c
            new_x[:,:pad_size,:] = right_x
            new_x[:,pad_size:-pad_size,:] = img
            new_x[:,-pad_size:,:] = left_x
    
            return  new_x, lab
