import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
#feature_map_length = 32  ## Hpw many of the 64 feature maps are we wantin g to look at]
# Gradient ascent
#steps = 1000  ## 250
#lr = .005    ## 0.01
#act_wt = 0.5 # factor by which to weigh the activation relative to the regulizer terms

class GradAscent_layers:
    def __init__(self, model_name, res, dir_pkl, pkl_name, device, batch_size =64, layre_num=0, feature_map_length=32, step=2, steps=1000, lr=0.0055, save=False, seed=None):
        """
        This function was adapted from a medium tutorial : https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
        Some aspects are exact copies such as gthe RGBgradients class
        """
        from modelCards import Cards
        
        self.model_name = model_name
        self.res = res
        self.dir_pkl = dir_pkl
        self.pkl_name = pkl_name
        self.device = device
        self.batch_size = batch_size
        self.layre_num = layre_num
        self.feature_map_length = feature_map_length
        self.step = step
        self.steps = steps
        self.lr = lr
        self.save = save
        self.seed= seed
    
        C = Cards()
        self.pad = C.res2pad(self.res)
        self.h = self.res[1]
        self.w = self.res[0]
        self.snow_img = torch.randn(self.batch_size, 3, self.h, self.w).to(self.device)
            
    def go(self):
        from modelManagment import load_pretrained_model
        self.model = load_pretrained_model(self.dir_pkl, self.pkl_name, self.model_name, self.res)
        if self.model_name == 'vgg16':
            print(self.model[0])
            self.model = self.model[0].to(self.device)
        else:
            self.model = self.model.conv_layers.to(self.device)

        filter = self.get_sharr_filter()
        gradLayer = self.gradrgb(filter)
        self.make_gradAscent() # l, ldx = 
        #self.view_gradascent(l, ldx)
    
    def gradrgb(self, weight):
        class RGBgradients(nn.Module):
            def __init__(self, weight): # weight is a numpy array
                super().__init__()
                k_height, k_width = weight.shape[1:]
                # assuming that the height and width of the kernel are always odd numbers
                padding_x = int((k_height-1)/2)
                padding_y = int((k_width-1)/2)
    
                # for each in_channel we have 2 out_channels corresponding to the x and the y gradients
                self.conv = nn.Conv2d(3, 6, (k_height, k_width), bias = False, 
                                      padding = (padding_x, padding_y) )
                weight1x = np.array([weight[0], 
                                     np.zeros((k_height, k_width)), 
                                     np.zeros((k_height, k_width))]) # x-derivative for 1st in_channel
                
                weight1y = np.array([weight[1], 
                                     np.zeros((k_height, k_width)), 
                                     np.zeros((k_height, k_width))]) # y-derivative for 1st in_channel
                
                weight2x = np.array([np.zeros((k_height, k_width)),
                                     weight[0],
                                     np.zeros((k_height, k_width))]) # x-derivative for 2nd in_channel
                
                weight2y = np.array([np.zeros((k_height, k_width)), 
                                     weight[1],
                                     np.zeros((k_height, k_width))]) # y-derivative for 2nd in_channel
                
                
                weight3x = np.array([np.zeros((k_height, k_width)),
                                     np.zeros((k_height, k_width)),
                                     weight[0]]) # x-derivative for 3rd in_channel
                
                weight3y = np.array([np.zeros((k_height, k_width)),
                                     np.zeros((k_height, k_width)), 
                                     weight[1]]) # y-derivative for 3rd in_channel
                
                weight_final = torch.from_numpy(np.array([          weight1x, weight1y, 
        weight2x, weight2y,
        weight3x, weight3y])).type(torch.FloatTensor)
            
                if self.conv.weight.shape == weight_final.shape:
                    self.conv.weight = nn.Parameter(weight_final)
                    self.conv.weight.requires_grad_(False)
                else:
                    print('Error: The shape of the given weights is not correct')
    
            def forward(self, x):
                return self.conv(x)
        gmod = RGBgradients(weight)
        return gmod


    
    def get_sharr_filter(self):
        # this filter is from https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
        filter_x = np.array([[-3, 0, 3], 
                             [-10, 0, 10],
                             [-3, 0, 3]])
        filter_y = filter_x.T
        return np.array([filter_x, filter_y])
    
    def grad_loss(self, beta = 1):
        # this loss function is from https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
        gradLayer.to(self.device)
        gradSq = gradLayer(self.img)**2 
        grad_loss = torch.pow(gradSq.mean(), beta/2)
        return grad_loss
    
        
    def make_gradAscent(self):
        relevant_layers = []
        # Hook function
        print("Creating Hook...")
        def hook_fn(module, input, output):
            global layer_output
            layer_output = output
            
        for layer in self.model:
            if isinstance(layer, torch.nn.Conv2d):
                relevant_layers.append(layer)
        
        for ldx, layer in enumerate(relevant_layers): # something to do with layer_num
            print(f"Layer         : {layer}")
            snow_ascent = []
            for channel_idx in range(0,self.feature_map_length, self.step): 
                #print(f"feature_map_length {feature_map_length}")
                #print(f"Channel     : {channel_idx}")
        
                snow = self.snow_img[0].clone().detach().requires_grad_(True)   ### requiring grad here is what is giving the 'backwards for the 2nd time' issue/
                
                handle = layer.register_forward_hook(hook_fn)
                #print("Hooked")
                #print(f"Hook count after registration: {len(layer._forward_hooks)}")
        
                self.model.eval()
        
                """feature_maps is a tensor that contains the activation maps of the final convolutional layer of the VGG16 model."""
                """Update the input_noise image by adding the gradient scaled by the learning rate lr."""
                
                optimizer = torch.optim.Adam([snow], lr=self.lr)
                #print(f"sending image through model...")
                for i in range(self.steps):
                    optimizer.zero_grad()
                    
                    _  = self.model(snow)
                    feature_maps = layer_output.squeeze_().requires_grad_().to(self.device)
        
                    loss = -feature_maps[channel_idx,:,:].mean()
                    loss.backward()
                    
                    optimizer.step()
                    
                #print(f'layer: {layer}, feature_num: {channel_idx}')
                input_noise_display = snow.detach().to('cpu').squeeze().permute(1, 2, 0)
                input_noise_display = torch.clamp(input_noise_display, 0, 1) # normalistion
                snow_ascent.append(input_noise_display)
                
        
                #for handle in hooks.values():
                handle.remove()
                #print(f"Hook count after removal: {len(layer._forward_hooks)}")
        
                if len(layer._forward_hooks) > 0:
                    print('more hooks than expected. something is wrong, restart the kernel.')
                #return snow_ascent, ldx
                #print(f"Your photos will be developed shortly for layer {ldx}...")
                    
            #def view_gradascent(self, snow_ascent, ldx):
            fig= plt.figure(figsize=(10, 6))
            plt.title(f"{self.model_name} Gradient Ascent Layer {ldx}")
            plt.axis("off")
            #rows, cols = int(np.floor((feature_map_length/step)/2)), int(np.ceil((feature_map_length/step)/2))
            print(f"len snow ascent {len(snow_ascent)}")
            cols = 5
            rows = int(np.ceil(len(snow_ascent)/ cols))
            for i in range(1, len(snow_ascent)):
                fig.add_subplot(rows, cols, i)
                plt.axis('off')
                plt.imshow(snow_ascent[i])
                if self.save:
                    plt.savefig(f"/its/home/nn268/antvis/antvis/optics/DeepDream/deepdream/2cInvestPics/{self.model_name}_{self.res}_{layer}_{self.seed}.jpg")
