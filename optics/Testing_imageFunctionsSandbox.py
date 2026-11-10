
import numpy as np
import cv2
from PIL import Image
import random

"""
This file is a testing script to assess the gauss_label function.
The gauss_label function creates a partial credit label where the True Label position 
  has a value of 1, and 22 positions (wrapped. circular) to either side contain values of
  decreaseing size towards 0.
  i.e.  0,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1 ,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,0
"""

w = 226 # width
h= 72 # height
block = np.zeros((h,w,3), dtype=np.uint8) # array of specified size



def gauss_label(north):
    """
    A function to produce a 360 degree label with a gaussian distribution of positive values (0-1) centered around 'north'.
    half cliprange : number of degrees covered by gaussian distribution * 2 (left and right of peak).
    std_dev : shape of guassian distribution (higher value is a wider curve, lower is sharper)#10925- opposite way around.
    """
    from scipy.signal.windows import gaussian

    num_degrees = 360
    if north <=0:
        north = 360 + north # always positive
    
    label = np.zeros(num_degrees, dtype='float32')   ## flaot32 here to match the float32 of the input images.
    half_ciprange = 23
    if half_ciprange%2 == 1:
        evenmaker = 1
    else:
        evenmaker = 0
    filtersize = (half_ciprange*2)+evenmaker
    std_dev = 7
    gauss = gaussian(filtersize, std_dev)
    gauss /= gauss.max() # normalise 

    ### bendy straights version (poker)
    for i, value in enumerate(gauss):
        index = (north - half_ciprange + i) % num_degrees
        label[index] = value
    
    return label


def colour_size_tense(image, vg =False):
        imNorm = None
        pixelOffset = None
        h = image.shape[0]
        print(f"h {h}")
        w = image.shape[1]
        image = cv2.resize(image, (360, h))
        
        #img = cv2.imread(image)
        im = image
        LorR = np.random.randint(2)
        print("LoR:",LorR)
        #print(f"LoR : {LorR}")
        pixelOffset = random.randint(0,int(im.shape[1]/2))  # For random rotation shifting # prod 1 or 0
        print(f"pixelOffset     {pixelOffset}")
        # image rotation is a randomly generated value between 0 and half image width
        
        if LorR == 1:
            
            #im = self.yaw(img, pixelOffset)
            # calc a label using rotation value
            # N is center of image (180)
            # += offset value
            label = gauss_label(180+pixelOffset)
        else:
            #im = self.yaw(img, -pixelOffset)
            label = gauss_label(180-pixelOffset) #label = self.gauss_label(self.labels[idx])
        #print(f" pixel offset of  {pixelOffset}  {im.shape}")
        #im = cv2.resize(im, (self.res[0], self.res[1]))
        
        #if vg:
        #    im = self.blank_padding(im, self.av_lum, (224,224)) 
        imNorm = im#/255 #norm
        print(f"max label:   {label.argmax()}")
        #imNorm[:,label.argmax()] = [0, 0, 255]
        print(imNorm.shape)
        print(f"pixelOffset     {pixelOffset}")
        imNorm[:,pixelOffset] = [0, 0, 255]
        
        # new_x[half_delta_h:-(half_delta_h+1),half_delta_w:-half_delta_w,:] = img
        #imNorm[:,pixelOffset] = (0, 0, 255)
        # max_index = my_list.index(max(my_list))
        imNorm = cv2.resize(imNorm, (w, h))
        
        print(imNorm.shape)
        #tense = self.to_tensor(imNorm) 
        tense= None
        
        return tense, label, im, imNorm



"""print(random.randint(0,int(block.shape[1]/2)))
print(random.randint(0,int(block.shape[1]/2)))
print(random.randint(0,int(block.shape[1]/2)))
print(random.randint(0,int(block.shape[1]/2)))
print(random.randint(0,int(block.shape[1]/2)))"""


_, _, _, IN = colour_size_tense(block)
_, _, _, imNorm2 = colour_size_tense(block)
_, _, _, imNorm3 = colour_size_tense(block)

imNorm = Image.fromarray(IN, mode='RGB')
imNorm.show()

imNorm2 = Image.fromarray(imNorm2, mode='RGB')
imNorm2.show()

imNorm3 = Image.fromarray(imNorm3, mode='RGB')
imNorm3.show()