# model managment functions
import torch
import torch.nn as nn
from torchvision.models import vgg16
from architectures import sevennet, eightnnet, sixnet, smallnet1, smallnet2, smallnet3
import pickle

#  SELECT AND INIT MODEL VIA  MODEL NAME (STR) #  SELECT AND INIT MODEL VIA  MODEL NAME (STR)
def choose_model(model_name, lin_lay, dropout, output_lin_lay=11):
    """
    Takes in model_name, lin_lay vlaue (size of input to the model head) and drop out values 
    Returns model with specified size
    """
    if model_name == '4c3l':
        return smallnet1(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '3c2l':
        return smallnet2(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks = (3,5), dropout=dropout)
    elif model_name == '2c2l':
        return smallnet3(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    if model_name == '6c3l':
        return sixnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '7c3l':
        return sevennet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == '8c3l':
        return eightnnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=output_lin_lay, ks= (3,5), dropout= dropout)
    elif model_name == 'vgg16':
        model_vgg16 = vgg16()
        vgg_classifier = model_vgg16.classifier
        vgg_classifier.pop(6) # removes output layer of VGG16 model
        vgg_mod = nn.Sequential(
            model_vgg16.features,
            nn.Flatten(),
            vgg_classifier,
            nn.Linear(4096,output_lin_lay), # Add new output layer appropriate to task
            nn.Softmax(dim=0),  
            )
                
        return vgg_mod
    else:
        print('Model Name Not Recognised')

def choose_model2(model_name, lin_lay, dropout): 
    # this version uses an imported vgg16 model [No Weights] with a custom output linear layer. 
    if model_name == '4c3l':
        return smallnet1(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '3c2l':
        return smallnet2(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks = (3,5), dropout=dropout)
    elif model_name == '2c2l':
        return smallnet3(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    if model_name == '6c3l':
        return sixnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '7c3l':
        return sevennet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '8c3l':
        return eightnnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == 'vgg16':
        #self.flatten = nn.Flatten()
        model_vgg16 = vgg16()
        vgg_feats = model_vgg16.features
        vgg_classifier = model_vgg16.classifier
        vgg_classifier.pop(6)

        vgg = nn.Sequential(
            vgg_feats,
            nn.Flatten(),
            vgg_classifier,
            nn.Linear(4096,11), # cheanging the output layer
            nn.Softmax(dim=0),  
            )
        return vgg
    else:
        print('Model Name Not Recognised')


def get_lin_lay(model_card, resolution):
    """
    Takes in model_card and specified resoltion
    Returns the corresponing classification head input size
    Model card is a dictionary
    Resolution is a key
    f_lin_lay is a value
    """
    if resolution == [452, 144]:
        lin_lay = model_card['f_lin_lay'][0]
    elif resolution == [226, 72]:
        lin_lay = model_card['f_lin_lay'][1]
    elif resolution == [113, 36]:
        lin_lay = model_card['f_lin_lay'][2]
    elif resolution == [57, 18]:
        lin_lay = model_card['f_lin_lay'][3]
    elif resolution == [29, 9]:
        lin_lay = model_card['f_lin_lay'][4]
    elif resolution == [15, 5]:
        lin_lay = model_card['f_lin_lay'][5]
    elif resolution == [8, 3]:
        lin_lay = model_card['f_lin_lay'][6]
    else:
        print("PARAMETER NOT FOUND: \n f_lin_lay FROM MODEL CARD")
    return lin_lay



def load_pretrained_model(dir_pkl, pkl_name, model_name, res:str):
    """
    Takes model and resolution -> to return model architecture of correct size for model_state_dict
    Loads model architecture
    Retrieves pickle file containing a model_state_dict
    Loads model_state_dict into model
    Returns model with pretrained weights
    """
    from modelCards import Cards
    C = Cards() 
    
    linlay = C.modname2linlay(model_name, res)
    model = choose_model(model_name, linlay, dropout=0.2) #
    print("Loading Weights into model...")
    try:
        with open(dir_pkl+pkl_name, 'rb') as f:
            model_pkl = torch.load(f)
            print("torch load")
    except:
        with open(dir_pkl+pkl_name, 'rb') as f:
            model_pkl = pickle.load(f)
            print("pkl load")
            
    model.load_state_dict(model_pkl['model.state_dict'])
    print("Done")
    return model


def choose_scheduler(save_dict, optimizer):
    """
    Returns learning rate scheduler
    Expects save_dict to include 'scheduler' item
    Expects 'scheduler' item to be a string of one of 'RoP', 'Exp' or 'NoSched'
    Returns corresponding scheduler or None if 'NoSched' is passed in
    """
    import torch.optim.lr_scheduler as lr_scheduler
    if save_dict['scheduler'] == "RoP":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=10)
    elif save_dict['scheduler'] == "Exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose='deprecated') 
    elif save_dict['scheduler'] == "NoSched":
        scheduler = None
    return scheduler
