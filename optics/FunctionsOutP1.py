import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

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
        elif (self.model_name == '8c3l' and size == [57, 15]) or (self.model_name == '8c3l' and size == [29, 9]) or (self.model_name == '8c3l' and self.res == [15, 5]) or (self.model_name == '8c3l' and size ==[8, 3]):
            tense = self.colour_size_tense(self.img_path[idx], vg=True)
            
        elif (self.model_name == '7c3l' and size == [29, 9]) or (self.model_name == '7c3l' and self.res == [15, 5]) or (self.model_name == '7c3l' and size ==[8, 3]):
            tense = self.colour_size_tense(self.img_path[idx], vg=True)
        elif (self.model_name == '6c3l' and self.res == [15, 5]) or (self.model_name == '6c3l' and size ==[8, 3]): #and size == [29, 9]) or (self.model_name == '6c3l'
            tense = self.colour_size_tense(self.img_path[idx], vg=True)
        else:
            tense = self.colour_size_tense(self.img_path[idx])        
        label = self.label_oh_tf(self.labels[idx])
        return tense, label