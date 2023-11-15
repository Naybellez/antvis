# creating unit tests for allll code
import unittest

import cv2
import numpy as np
import sys 
#sys.path.insert(0,'..')
#sys.path.append("..")
import os

#print(os.path.abspath("../functions.py")) #/its/home/nn268/antvis/optics/functions.py

import importlib.util
spec= importlib.util.spec_from_file_location("functions", "functions.py")
functions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(functions)


#print('SYS PATH \n',sys.path)
# .clone()

ImageProcessor = functions.ImageProcessor
get_data = functions.get_data


# data processing functions
                            # importing data
                            # split data
                            # tensor labels

# image processing functions
#from functions import ImageProcessor, get_data

class Test_IP(unittest.TestCase):

    def setUp(self):
        IP_1 = ImageProcessor(device='cpu')
        self.x, self.y, _,_,_,_ = get_data(file_path=r'/its/home/nn268/antvis/optics/AugmentedDS_IDSW/')
        test_x, test_y = x[0], y[0]
        self.test_x= cv2.imread(test_x)
        self.test_grey = cv2.cvtColor(test_x, cv2.COLOR_BGR2GRAY)
    def tearDown(self):
        pass
    def test_1(self):
        
        # splitting channels
        r,g,b = self.IP_1.split_channels(self.test_x)
        # check all channels are teh same size
        self.assertEqual(r.shape, self.test_x[:,:,0].shape)
        self.assertEqual(b.shape, self.test_x[:,:,1].shape)
        self.assertEqual(g.shape, self.test_x[:,:,2].shape)
        self.assertIsNotNone(r)
        self.assertNotEqual(r, b)

        # check input type
        #self.assertFalse(self.IP_1.split_channels(self.test_y))
        #self.assertFalse(self.IP_1.split_channels(self.test_grey))
    




                            # splitting channels
                            # calc luminance
    def test_2(self): # luminance
        self.assertTrue(self.IP_1.luminance(self.test_x))
        
        
    
    def test_3(self):
        self.assertCountEqual(self.x,self.y)

                            # creating padding
                            # tensoring image
                            # viewing image post processing

# loop functions

# 

if __name__ == '__main__':
    unittest.main()