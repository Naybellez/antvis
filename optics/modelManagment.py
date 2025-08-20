# model managment functions
import torch
import torch.nn as nn
from architectures import PrintLayer#, Flattern
from architectures import sevennet, eightnnet, sixnet, smallnet1, smallnet2, smallnet3
from torchvision.models import vgg16
import torch.nn.functional as F
import pickle

#  SELECT AND INIT MODEL VIA  MODEL NAME (STR) #  SELECT AND INIT MODEL VIA  MODEL NAME (STR)
def choose_model(model_name, lin_lay, dropout, output_lin_lay=11):
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
        from torchvision.models import vgg16
        model_vgg16 = vgg16()
        vgg_classifier = model_vgg16.classifier
        vgg_classifier.pop(6)
        vgg_mod = nn.Sequential(
            model_vgg16.features,
            nn.Flatten(),
            vgg_classifier,
            nn.Linear(4096,output_lin_lay), # cheanging the output layer
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


def choose_model_out10(model_name, lin_lay, dropout): 
    # this version uses an imported vgg16 model [No Weights] with a custom output linear layer. 
    if model_name == '4c3l':
        return smallnet1(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '3c2l':
        return smallnet2(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks = (3,5), dropout=dropout)
    elif model_name == '2c2l':
        return smallnet3(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
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
            nn.Linear(4096,10), # cheanging the output layer
            nn.Softmax(dim=0),  
            )
        return vgg
    else:
        print('Model Name Not Recognised')


def choose_model1(model_name, lin_lay, dropout=0.2):
    # this version creates vgg16 layer by layer, NOT an imported model
    if model_name == '4c3l':
        return smallnet1(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '3c2l':
        return smallnet2(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks = (3,5), dropout=dropout)
    elif model_name == '2c2l':
        return smallnet3(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '6c3l':
        return sixnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '7c3l':
        return sevennet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == '8c3l':
        return eightnnet(in_chan=3, f_lin_lay=int(lin_lay), l_lin_lay=11, ks= (3,5), dropout= dropout)
    elif model_name == 'vgg16':
        class VGG16Smaller(nn.Module):
            def __init__(self,lin_lay=200704, num_classes=11): #64512
                super(VGG16Smaller, self).__init__()
                self.layer1 = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU()),
                self.layer2 = nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(), 
                    nn.MaxPool2d(kernel_size = 2, stride = 2))
                self.layer3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
                self.layer4 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2))
                self.layer5 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()),
                self.layer6 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU()),
                self.layer7 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size = 2, stride = 2))
                self.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(lin_lay, 4096), # 1032192 and 4096x4096)
                    nn.ReLU()),
                self.fc1 = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU()),
                self.fc2= nn.Sequential(
                    nn.Linear(4096, num_classes))
                
            def forward(self, x):
                out = self.layer1(x)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.layer5(out)
                out = self.layer6(out)
                out = self.layer7(out)
                PrintLayer()
                out = out.reshape(out.size(0), -1)
                out = out.flatten(start_dim=1)
                PrintLayer()
                out = self.fc(out)
                out = self.fc1(out)
                out = self.fc2(out)
                out = F.log_softmax(out, dim=1) 
                return out
        vgg = VGG16Smaller(lin_lay)
        return vgg
    else:
        print(f'Model Name Not Recognised : {model_name}')

def get_lin_lay(model_card, resolution):
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
    #from modelManagment import choose_model#1
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
    import torch.optim.lr_scheduler as lr_scheduler
    if save_dict['scheduler'] == "RoP":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=10)
    elif save_dict['scheduler'] == "Exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1, verbose='deprecated') 
    elif save_dict['scheduler'] == "NoSched":
        scheduler = None
    return scheduler
