# Script contains a class for creating ClassActivationMapping images
# Following https://github.com/jacobgil/pytorch-grad-cam
"""
@misc{jacobgilpytorchcam,
  title={PyTorch library for CAM methods},
  author={Jacob Gildenblat and contributors},
  year={2021},
  publisher={GitHub},
  howpublished={\url{https://github.com/jacobgil/pytorch-grad-cam}},
}
"""

# Imports
import torch
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../../.')

class CAMfromPickle():
    def __init__(self, model_file, directory, data_path, brightness=False, skyblock=False, try_pkl=True, twotone=False, horzflip=False, vertflip=False): 
        """
        takes in 
        model_file : model weights and simulation parameters
        directory : file path to model_file
        data_path : file path to image data

        Initialise with Augmentation flags set
        All false flags equals no additional image augmentations
        brightness : brightness inversion
        skyblock : adds three colourful rectangles to the upper two thirds of the image, blocking the skyline panormama
        twotone : attemps to reduce a 3 channel image into a two channel image (close to black and white)
        horizflip : flips the image along the horizontal plane
        vertflip : flips the image along the vertical plane

        try_pkl : whether to use pickle.load or torch.load to load model weights (held in pkl file)
        """
        import torch
        import pickle
        import ast
        
        print('initialising...')
         # 
        print('loading model...')

        if try_pkl:
            with open(directory+model_file, 'rb') as f:
                model_save_dict = pickle.load(f)
        else:
            with open(directory+model_file, 'rb') as f:
                model_save_dict = torch.load(f)

        
        #print('Model: ', list(model_save_dict)) # unhash to print the full dictionary

        # retreive simulation parameters from model_save_dict
        self.model_name = model_save_dict['model_name']
        self.linear_layer = model_save_dict['lin_lay']
        self.learning_rate = model_save_dict['lr']
        self.epochs_trained = model_save_dict['Current_Epoch']
        self.model_state_dict = model_save_dict['model.state_dict']
        self.seed = int(model_save_dict['seed'])
        self.resolution = ast.literal_eval(model_save_dict['resolution'])
        self.pad = model_save_dict['pad']
        self.drop_out = 0.2 

        self.data_path = data_path
        self.directory = directory
        print("initialised")
        self.skyblock = skyblock
        self.brightness = brightness
        self.twotone = twotone
        self.horzflip = horzflip
        self.vertflip = vertflip
        
    def get_testset(self):
        # Retrieves data, preprocessing and dataloader initiation
        from functions import get_data, IDSWDataSetLoader9, ImageProcessor 
        from torch.utils.data import DataLoader
        print("getting data ready...")
        
        x_train, _, _, _, x_test, y_test = get_data(file_path=self.data_path,random_seed=self.seed)
        self.y_test = x_test

        IP = ImageProcessor(device='cpu')
        av_lum = IP.new_luminance(x_train) # calc average luminance of training set to be used as balnk padding (added in dataloader preprocessing)
        test_ds = IDSWDataSetLoader9(x_test, y_test, self.resolution,self.pad,av_lum,model_name=self.model_name,
                                      device='cpu', skyblock=self.skyblock, b_invert=self.brightness, twotone=self.twotone,
                                        horzflip=self.horzflip, vertflip=self.vertflip)
        self.test = DataLoader(test_ds, batch_size=64, shuffle=True, drop_last=False) 
        print("test data is loaded.")

    def fill_model(self, model):
        # Load pretrained weights into model
        print("filling model...")
        self.model = model
        self.model.load_state_dict(self.model_state_dict)
        print("model filled")

    
    def trans_to_img(self, img, scale):
        # Transform torch tensor to numpy array for image viewing
        # useful for checking if image augmentations applied correctly
        import numpy as np
        if isinstance(img, torch.Tensor):
            if len(img.shape) ==4:
                img = img.squeeze()
                img = img.permute(2,3,1,0)
                img = np.array(img.cpu())*scale
            elif len(img.shape)==3:    
                img = img.squeeze()
                img = img.permute(1,2,0)
                img = np.array(img.cpu())*scale
            else:
                print("image shape not recognised", img.size, img.shape, len(img.shape))

        elif isinstance(img, np.ndarray):
            img = img*scale
        elif isinstance(img, str): 
            img = cv2.imread(img)
        return img


    def createCAMfig(self, save=True):
        # handles CAM image creation  using the test set
        import matplotlib.pyplot as plt
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        img_names = []
        if self.model_name== 'vgg16': 
            target_layer = self.model[0]
        else:
            target_layer = self.model.conv_layers 
                
        self.model.eval()
        num_correct = 0
        total_seen = 0
        
        for idx, batch in enumerate(self.test):
        
            tense, label = batch
            label = label.to('cpu')
            out = self.model.forward(tense.to('cpu'))
            for jdx in range(len(label)): # get accuracy count
                if out[jdx].argmax() == label[jdx].argmax():
                    num_correct +=1
            total_seen +=len(label)

            # initialise GradCAM object
            Cam = GradCAM(self.model, target_layer)
            targets = None
            camout = Cam(tense, targets) 
            for idx, img in enumerate(tense):

                img = self.trans_to_img(img, 1) # array of image for visualisation
                img_names.append(img)

                visualisation = show_cam_on_image(img, camout[idx], use_rgb=True)
                f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,3)) #8, 4
                ax1.imshow(visualisation)
                ax1.axis(False)
                ax2.imshow(img)
                f.suptitle(f"{self.model_name} Resolution {self.resolution} \n Location: {label[idx].argmax()}   Predicted: {out[idx].argmax()}", y=1.2)
                plt.axis(False)

                # choosing subfolder by augmentation applied
                import os
                if self.brightness==True:
                    subfolder="bright"
                elif self.skyblock ==True:
                    subfolder="skyblock"
                elif self.twotone == True:
                    subfolder="twotone"
                elif self.horzflip == True:
                    subfolder="horzflip"
                elif self.vertflip == True:
                    subfolder = "vertflip"
                else:
                    subfolder="clean"
                if save:
                    parent_dir = self.directory
                    child_dir = f"{self.model_name}/cam/{self.resolution}/{subfolder}/"
                    os.chdir(parent_dir)
                    if not os.path.exists(parent_dir+child_dir):
                        os.makedirs(parent_dir+child_dir)
                    image_name = f"Loc:{label[idx].argmax()}_Pred:{out[idx].argmax()}_({idx})"
                    plt.savefig(parent_dir+child_dir+image_name+".png")
                plt.show()

        print("Test accuracy: ", (num_correct / total_seen)*100)
        print(num_correct, total_seen)


class CamDoubleplot():
    #  cam images already made
    # this creates a two column image plot with the original on the left and CAM on the right
    def __init__(self, image_folder, saveto_folder, res, test = 'clean'):
        self.image_folder = image_folder
        self.saveto_folder= saveto_folder
        self.res = res
        self.test= test
        self.cor_per_res_samples = []
        # I want this lists to contain the randomly selected files, with their full file path attached- the saved file path
        self.incor_per_res_samples = []

        correct_list, incorrect_list = self.sortCAMfiles()
        print("cor example: ", correct_list[0])
        print("incor example: ", incorrect_list[0])
        self.correct_samples = self.get_samples(correct_list, var="correct")
        self.incorrect_samples = self.get_samples(incorrect_list, var="incorrect")

        print("sample example:     ",self.correct_samples[0])
        
        self.save_sample('correct'+str(self.test), self.correct_samples)
        self.save_sample('incorrect'+str(self.test), self.incorrect_samples)
        
    
    def sortCAMfiles(self):
        # Looks in folder of CAM images
        # uses image name to sort files as correct or incorrect predcitions
        # returns sorted lists
        correct_list = []
        wrong_list = []
        
        loc = None
        pred = None
        
        for im_file_name in os.listdir(self.image_folder): # iterate though files in directory
            if im_file_name.lower().startswith('loc'): # proceed if 'loc' code present
                if not im_file_name[4:6].isnumeric(): # lcoaiton code, single digit - 1, 2, 3...
                    loc = im_file_name[4]
                    i = 0
                else:
                    loc = im_file_name[4:6] # retrive location code. double digits - 10, 11
                    i = 1
                if not im_file_name[11+i:13+i].isnumeric(): # retrive prediction from file name
                    pred = im_file_name[11+i]
                elif im_file_name[11+i:13+i].isnumeric():
                    pred = im_file_name[11+i:13+i]
                if int(loc) == int(pred): # checks for match betwen location code and prediciton
                    correct_list.append({im_file_name:[loc]})
                else:
                    wrong_list.append({im_file_name: [loc, pred]}) 
        return correct_list, wrong_list

    def get_samples(self, im_list, var=None):
        # get random selection of samples
        # takes in im_list : list of images sorted as correct or incorrectly predicted
        #          var : string - 'correct'/'incorrect'
        # returns sample_out : set size. random selected of images.
        sample_size = 0
        sample_out = []
        
        max_sample = min(len(im_list), 20) # Max number of samples set here to 20
        print(f"Max length of {var} sample list:  ",max_sample)
        while sample_size < max_sample:
            random_num = random.randint(0, len(im_list)-1)
            sample_out.append(im_list[random_num])
            sample_size +=1
        return sample_out 

    # save image code
    def save_sample(self, sorted, sample_list):
        # takes in sorted : string of correct/incorrect and augmentation name
        #          sample_list : list of samples sorted as either correct / incorrect
        # saves and plots the heatmapped image
        for im in sample_list:
            im_key = list(im.keys())[0]
            img = Image.open(self.image_folder +"/"+im_key)
            self.saveIm(img, im_key, sorted)
            plt.imshow(img)
            plt.grid(False)
        
    def saveIm(self, image, image_name, sorted): # sorted - correct/ incorrect
        # Function handles saving of image data
        if image_name[-4:] != '.png':
            image_name+'.png'
            
        self.child_dir = f"/randomCAMs/" # random slection of CAM images
        sorted_dir = f"{sorted}/"

        # check if save path exists, create if not
        if not os.path.exists(self.saveto_folder+self.child_dir+sorted_dir):
            os.makedirs(self.saveto_folder+self.child_dir+sorted_dir)
            
        image.save(self.saveto_folder+self.child_dir+sorted_dir+image_name)

    def Choose_rand_subsample(self):
        # we have a list of 20
        # We want to choose one to return - full file path
        print(len(self.correct_samples))
        max = min(len(self.correct_samples), len(self.incorrect_samples)) # sample limit cannot be greater than number of raw samples
        random_index = random.randint(0, max-1)
        
        selected_corr_im = list(self.correct_samples[random_index].keys())[0] # use random_index to retrive a correct sample
        selected_incorr_im = list(self.incorrect_samples[random_index].keys())[0] # use random_index to retrive an incorrect sample
        print("selected cor key", selected_corr_im) # file name
        print("self test", self.test) # augmention type

        selected_corr_out = str(selected_corr_im) # file path must be string
        selected_incorr_out = str(selected_incorr_im)
        cor_path = self.saveto_folder+self.child_dir+'correct'+str(self.test)+"/" # Directory path Correct
        incor_path = self.saveto_folder+self.child_dir+'incorrect'+str(self.test)+"/" # Directory path Incorrect
        
        return cor_path+selected_corr_out, incor_path+selected_incorr_out # return full paths

        
def crop_im(im):
    # Crops input image to set size
    # Returns cropped image
    w, h = im.size
    im.show()
    left = 95
    top = h/4 - 100#165#h +20
    right = 725
    bottom = 3*h/3 -5
    
    cropped_im = im.crop((left, top, right, bottom))
    cropped_im.show()
    return cropped_im

def crop_cams(all_samples):
    # Handles iterating through input list 'all_samples'
    # applies function crop_im to each image
    # Returns a list of cropped images
    read_samples = []
    for i in all_samples:
        k = Image.open(i)
        k = crop_im(k)
        read_samples.append(k)
    return read_samples


def multiCAMplot(cam_samples, test_type:str, model_type:str, w_space = -0.45, h_space=0.1):
    # This function creates a multiplot of CAMDoubeplot images
    # correctly classified are on the left in green, incorrectly on the right with red
    # set the column heading names
    cols = ['Correctly Classified', 'Incorrectly Classified']
    # set the row order
    rows = ["Resolution {}".format(row) for row in ['[452, 144]', '[226, 72]', '[113, 36]', '[57, 18]', '[29, 9]', '[15, 5]', '[8, 3]']]
    # create a blank multiplot with 7 rows and 2 columns. gridspec is used to control spacing
    f, axarr = plt.subplots(7, 2, figsize=(8, 12), gridspec_kw = {'wspace':w_space, 'hspace':h_space}) #, gridspec_kw = {'wspace':0.0005, 'hspace':0.0002}  figsize=(8, 12), gridspec_kw = {'wspace':-0.45, 'hspace':0.1}
    
    def plot_image(ax, img):
        # mini function to plot the images
        ax.imshow(img) 
        ax.grid(False)
        ax.axis('off')
    # loop to plot images in correct row column axis
    for i in range(7):
        for j in range(2):
            plot_image(axarr[i, j], cam_samples[i*2 +j])
    # give the column headings oretty colours
    axarr[0,0].set_title(cols[0], fontsize=10, pad=5, color='#008000')
    axarr[0,1].set_title(cols[1], fontsize=10, pad=5, color= '#CC79A7')
    
    for ax, row in zip(axarr[:,0], rows):
        ax.set_ylabel(row, rotation=0,  ha='right', va='center') # size=10,
    
    plt.suptitle(f"{model_type} {test_type} CAM Images \n Illistrating Areas of Interest \n for Location Classification")
    
    plt.tight_layout()
    plt.savefig(f'CAM_{model_type}_{test_type}.png', dpi=300, bbox_inches='tight')
    plt.show()