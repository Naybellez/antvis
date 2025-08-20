import torch
import numpy
import matplotlib.pyplot as plt
import pickle


def print_mod_layers(model):
    """this function is written for custom models with conv layers in 'conv_layers' or vgg16 set in sequential blocks"""
    for name, module in model.named_children():
        print(name, type(name))
        if name == 'conv_layers' or name == 0:
            print(name , module)
            for layer in module:
            #print(model)
                if isinstance(layer, torch.nn.Conv2d):
                        print(layer)


def get_conv_layers(model):
    """This function is written for custom models with conv layers in 'conv_layers' or vgg16 set in sequential blocks"""
    lays = []
    for name, module in model.named_children():
        if name == 'conv_layers' or name == '0':
            for layer in module:
                if isinstance(layer, torch.nn.Conv2d):
                    print(layer)
                    lays.append(layer)

    # Return the actual layer modules, not just their weights
    layers = [l.weight for l in lays]  # Keep this for backward compatibility if needed
    f_min, f_max = layers[0].min(), layers[0].max() if layers else (0, 1)
    
    # Return: actual_layers, weights, f_min, f_max
    return lays, layers, f_min, f_max



def get_conv_params(model):
    """this function is written for custom models with conv layers in 'conv_layers' or vgg16 set in sequential blocks"""
    lays = []
    for name, module in model.named_children():
        if name =='conv_layers' or name =='0':
            for layer in module:
                if isinstance(layer, torch.nn.Conv2d):
                    print(layer)
                    lays.append(layer)


    layers = [l.weight for l in lays]
    f_min, f_max = layers[:][0].min(), layers[:][0].max()
    return lays, layers, f_min, f_max



def see_kernels2(layers: list):
    """this function creates a multiplot of cnn kernels. row per kernel, col per channel
    Run get_layers to get the indivual conv layers to use in this function
    function perfected with Claude AI"""
    for layer_idx, layer in enumerate(layers):
        print(f"Layer {layer_idx}")
        filters = layer.detach().numpy()
        n_filters, n_channels = filters.shape[0], filters.shape[1]
        
        # Create figure with n_filters rows, n_channels columns
        fig, axes = plt.subplots(n_filters, n_channels, 
                                figsize=(n_channels*2, n_filters*2))
        
        for i in range(n_filters):
            for j in range(n_channels):
                ax = axes[i, j] if n_filters > 1 else axes[j]
                ax.imshow(filters[i, j], cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                #ax.set_title(f'K{i+1}C{j+1}' if i == 0 else '')
                ax.set_title(f'K{i+1}-C{j}')
        
        plt.tight_layout()
        plt.show()



def featmap_advanced(featpred, max_maps=64, grid_cols=None):
    """
    This function creates a multiplot of conv feature maps.
    function perfected with Claude AI
    
    Args:
        featpred: Feature maps tensor/array of shape (n_maps, height, width)
        max_maps: Maximum number of feature maps to display
        grid_cols: Number of columns (if None, uses square grid)
    """
    n_maps = min(featpred.shape[0], max_maps)
    
    if grid_cols is None:
        # Square grid
        grid_cols = int(np.ceil(np.sqrt(n_maps)))
        grid_rows = grid_cols
    else:
        # Custom columns
        grid_rows = int(np.ceil(n_maps / grid_cols))
    
    # Create figure
    fig_width = max(8, grid_cols * 1)
    fig_height = max(6, grid_rows * 0.9)
    plt.figure(figsize=(fig_width, fig_height))
    
    for i in range(n_maps):
        ax = plt.subplot(grid_rows, grid_cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.imshow(featpred[i], cmap='gray')
        ax.set_title(f'FM {i}', fontsize=8)
    
    # Hide unused subplots
    total_subplots = grid_rows * grid_cols
    for i in range(n_maps, total_subplots):
        ax = plt.subplot(grid_rows, grid_cols, i + 1)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"Displayed {n_maps} out of {featpred.shape[0]} feature maps")



class ShowFeatmap:
    def __init__(self, res, dir_pkl, pkl_name, model_name, device, img=None, batch_size = 64, layer_num = 0):
        from  modelCards import Cards
        
        C = Cards()
        self.res = res
        self.dir_pkl = dir_pkl
        self.pkl_name = pkl_name
        self.model_name = model_name
        self.device = device
        self.img = img
        self.batch_size = batch_size
        self.layer_num = layer_num
        
        self.pad = C.res2pad(self.res)
        self.h = self.res[1]
        self.w = self.res[0]+self.pad
        
        if self.img ==None:
            self.img = torch.randn(self.batch_size, 3, self.h, self.w).to(self.device)
            
    def go (self):
        from modelManagment import load_pretrained_model
        self.model = load_pretrained_model(self.dir_pkl, self.pkl_name, self.model_name, self.res).to(self.device)
        self.model = self.model.conv_layers
        for param in self.model.parameters():
            param.requires_grad_(False)
        while self.layer_num <= len(self.model):#.conv_layers):
            l = self.get_featmap_activation()
            if l == None:
                #sys.exit(0) # -q # 'q', '-q'
                return print("Done")
            self.view_featmap_activations(l)
            self.layer_num +=1
        
    def get_convs(self):
            self.layer = None
            if self.layer_num >= len(self.model ): #.conv_layers
                return None
            if isinstance(self.model[self.layer_num], torch.nn.Conv2d): #conv_layers
                layer = self.model[self.layer_num] #conv_layers
                print(f"Conv layer {self.layer_num} isolated")
                return layer
            else:
                print("Trying next layer!")
                self.layer_num += 1
                return self.get_convs()
                
    def get_featmap_activation(self):
        import sys
        def hook_fn(module, input, output):
            nonlocal layer_output
            layer_output = output
            
        # get layer to add hook
        layer = self.get_convs()
        if layer == None:
            return print("Done")
        # get hooked
        handle = layer.register_forward_hook(hook_fn)
        print(f"Hook Registered for layer {self.layer_num}")
    
        # get the layer_output
        with torch.inference_mode():
            _ = self.model(self.img)
            layer_output = torch.squeeze(layer_output,dim=1) # layeroutput is from the hook
            print(f"Got Layer output")
        return layer_output
    
    # send snow through model
    def view_featmap_activations(self, layer_output):
        
        # Visualise
        #print("view featmap activs func   : ",layer_output.shape, layer_output.shape[0])
        totalfeats = int(layer_output.shape[1])
        cols = 5
        rows = int(np.ceil(totalfeats/cols))
        scale =  int(np.ceil(rows/cols))#max(1, rows/2)#int(rows / totalfeats)#rows/
        #print(f"scale   {scale}, totalfeats {totalfeats}, cols {cols}")
        #print(f"scale   {scale}")
        fig = plt.figure(figsize=(10, 6*scale))
        plt.title(f"{self.model_name} Feature Map. Conv Layer {self.layer_num}. {totalfeats} out of {len(layer_output[0])} out channels")
        plt.axis(False)
        
        for i in range(1, (rows * cols) + 1):
            if i >= len(layer_output[0]):
                continue
            feature_map = layer_output[0][i-1, :, :].cpu().numpy()
            fig.add_subplot(rows, cols, i)
            plt.imshow(feature_map, cmap='viridis')
            plt.tight_layout()
            plt.axis(False)
            
            
    



import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
#feature_map_length = 32  ## Hpw many of the 64 feature maps are we wantin g to look at]
# Gradient ascent
#steps = 1000  ## 250
#lr = .005    ## 0.01
#act_wt = 0.5 # factor by which to weigh the activation relative to the regulizer terms

def GradAscent_layers(model_name, res, dir_pkl, pkl_name, device, batch_size =64, layre_num=3, feature_map_length=32, step=2, steps=1000, lr=0.0055, rows=6, cols=6):
    """
    This function was adapted from a medium tutorial : https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
    Some aspects are exact copies such as gthe RGBgradients class
    """
    from modelCards import Cards
    from modelManagment import load_pretrained_model

    C = Cards()
    pad = C.res2pad(res)
    h = res[1]
    w = res[0]
    snow_img = torch.randn(batch_size, 3, h, w).to(device)
        
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

    def get_sharr_filter():
        # this filter is from https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
        filter_x = np.array([[-3, 0, 3], 
                             [-10, 0, 10],
                             [-3, 0, 3]])
        filter_y = filter_x.T
        return np.array([filter_x, filter_y])

    def grad_loss(img, beta = 1, device = device):
        # this loss function is from https://medium.com/@deepeshdeepakdd2/cnn-visualization-techniques-feature-maps-gradient-ascent-aec4f4aaf5bd
        gradLayer.to(device)
        gradSq = gradLayer(img)**2 
        grad_loss = torch.pow(gradSq.mean(), beta/2)
        return grad_loss

    
    def view_gradAscent(img_noise, rows, cols):
        relevant_layers = []
        # Hook function
        print("Creating Hook...")
        def hook_fn(module, input, output):
            global layer_output
            layer_output = output
            
        for layer in model:
            if isinstance(layer, torch.nn.Conv2d):
                relevant_layers.append(layer)
        
        for ldx, layer in enumerate(relevant_layers): # something to do with layer_num
            print(f"Layer         : {layer}")
            snow_ascent = []
            for channel_idx in range(0,feature_map_length, step): 
                #print(f"feature_map_length {feature_map_length}")
                #print(f"Channel     : {channel_idx}")
        
                snow = img_noise[0].clone().detach().requires_grad_(True)   ### requiring grad here is what is giving the 'backwards for the 2nd time' issue/
                
                handle = layer.register_forward_hook(hook_fn)
                #print("Hooked")
                #print(f"Hook count after registration: {len(layer._forward_hooks)}")
        
                model.eval()
        
                """feature_maps is a tensor that contains the activation maps of the final convolutional layer of the VGG16 model."""
                """Update the input_noise image by adding the gradient scaled by the learning rate lr."""
                
                optimizer = torch.optim.Adam([snow], lr=lr)
                #print(f"sending image through model...")
                for i in range(steps):
                    optimizer.zero_grad()
                    
                    _  = model(snow)
                    feature_maps = layer_output.squeeze_().requires_grad_().to(device)
        
                    loss = -feature_maps[channel_idx,:,:].mean()
                    loss.backward()
                    
                    optimizer.step()
                    
                #print(f'layer: {layer}, feature_num: {channel_idx}')
                input_noise_display = snow.detach().to('cpu').squeeze().permute(1, 2, 0)
                input_noise_display = torch.clamp(input_noise_display, 0, 1)
                snow_ascent.append(input_noise_display)
                
        
                #for handle in hooks.values():
                handle.remove()
                #print(f"Hook count after removal: {len(layer._forward_hooks)}")
        
                if len(layer._forward_hooks) > 0:
                    print('more hooks than expected. something is wrong, restart the kernel.')
            
            print(f"Your photos will be developed shortly for layer {ldx}...")
            fig= plt.figure(figsize=(10, 6))
            plt.title(f"{model_name} Gradient Ascent Layer {ldx}")
            plt.axis("off")
            #rows, cols = int(np.floor((feature_map_length/step)/2)), int(np.ceil((feature_map_length/step)/2))
            
            for i in range(1, len(snow_ascent)):
                fig.add_subplot(rows, cols, i)
                plt.axis('off')
                plt.imshow(snow_ascent[i])

    model = load_pretrained_model(dir_pkl, pkl_name, model_name, res)
    model = model.conv_layers.to(device)
    filter = get_sharr_filter()
    gradLayer = RGBgradients(filter)
    view_gradAscent(snow_img, rows, cols)