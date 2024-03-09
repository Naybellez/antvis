import os
import numpy as np
from sklearn.model_selection import train_test_split

from functions import import_imagedata, label_oh_tf, ImageProcessor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torchvision.models import vgg16
from torchvision.models import resnet101

import cv2

from torch.utils.data import DataLoader
import wandb
from fns4wandb import train_log, build_optimizer
from copy import deepcopy
from tqdm import tqdm

import random
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import pickle
from fns4wandb import set_lossfn




device = "cuda:0" if torch.cuda.is_available() else "cpu" #main


model_resnet = resnet101(weights="IMAGENET1K_V1")#.eval
newmodel = torch.nn.Sequential(*(list(model_resnet.children())[:-1]))
newmodel=newmodel.to(device)
print(newmodel) # get resnet model function





def preprocess_im(img_path):
    IP = ImageProcessor(device='cpu')
    img = cv2.imread(img_path) #
    img = IP.blank_padding(img, (224,224))
    img = IP.to_tensor(img).to(device)
    return img



title = f'IDSW_resnet_fine_120_112023'
save_dict = {'Run' : title,
            'Current_Epoch': 0}
                #r'/its/home/nn268/antvis/optics/
save_location = r'pickles/'#pickles




    



def train_model(model,  x_train, x_val, y_train, y_val, config, best_acc=0): #train_dl, val_dl, 
    #wandb.watch(model, log='all', log_freq=10)
    
    loss_fn = set_lossfn(config['loss_fn']) # ****
    
    lr = config['learning_rate'] #1e-5 #config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#build_optimizer(model, optimizer=torch.optim.Adam(model.parameters(), lr=lr))#config.optimizer, config.learning_rate, config.weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config['scheduler'], last_epoch=-1) #gamma=config.scheduler, last_epoch=-1)
    
    ####
    
    
    model.train()
    
    
    #losses= []
    #predictions = []
    t_loss_list = []
    v_loss_list = []
    t_predict_list = []
    v_predict_list = []
    t_accuracy_list = []
    v_accuracy_list = []
    t_label_list = []
    v_label_list = []
    #labels = []
    
    total_epochs = 0
    for epoch in tqdm(range(config['epochs'])): #config.epochs)):
        print('E   ', epoch)
        t_correct = 0
        v_correct = 0
    
        if epoch == 0:
            model = model.to('cpu')
            best_model = deepcopy(model)
            model = model.to(device)
            
            
        #train_ids = random.shuffle(train_ids)
        #print(type(train_ids))
        print('training...')
        for idx, img in enumerate(x_train): 
            model.train()

            x = preprocess_im(img)

            train_prediction = model.forward(x)
            
            y = label_oh_tf(y_train[idx], device=device, num_classes=11) # use same index val to index y (labels) and turn into onehot encoded label

            """
            if idx % 1000 == 0:
                print(idx, ' / ', len(x_train))
                !nvidia-smi
            """
            #print(train_prediction, train_label)
        
        
            t_loss = loss_fn(train_prediction, y)

            if train_prediction.argmax() == y.argmax():
                t_correct+=1

            
            optimizer.zero_grad()
            t_loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            t_loss_list.append(t_loss.to('cpu'))
            t_predict_list.append(train_prediction.to('cpu'))
            t_label_list.append(y.to('cpu'))
            
            train_acc = (t_correct / len(x_train))
            t_accuracy_list.append(train_acc)
            
            
            
        print('validating...')
        
        for idx, img in enumerate(x_val):
            model.eval()
            x = preprocess_im(img)

            val_prediction = model.forward(x)
            
            y = label_oh_tf(y_val[idx], device=device, num_classes=11)
            """
            if idx % 100 == 0:
                print(idx, ' / ', len(x_val))
                !nvidia-smi
            """

            v_loss = loss_fn(val_prediction, y)
            
            if val_prediction.argmax() == y.argmax():
                v_correct +=1
                

            v_loss = v_loss.to('cpu')
            v_loss_list.append(v_loss.item())
            
            
            val_prediction = val_prediction.to('cpu')
            v_predict_list.append(val_prediction.detach().numpy())
            
            v_label_list.append(y.to('cpu').detach().numpy())
            
            val_acc = (v_correct / len(y_val))
            v_accuracy_list.append(val_acc)
            
        total_epochs += epoch

        if val_acc > best_acc:
    
            #print('Start Of SAve -----------------------------')
            #!nvidia-smi

            best_acc = val_acc
            
            model = model.to('cpu')
            best_model = model#deepcopy(model)
            model = model.to(device)
            

            
            save_dict['Current_Epoch'] += config['epochs']
            save_dict['training_samples'] = len(x_train)
            save_dict['validation_samples'] = len(x_val)
            save_dict['t_loss_list'] = t_loss_list
            save_dict['t_predict_list'] = t_predict_list  
            save_dict['t_accuracy_list'] = t_accuracy_list  #
            save_dict['v_loss_list'] = v_loss_list
            save_dict['v_predict_list'] = v_predict_list  #
            save_dict['v_accuracy_list'] = v_accuracy_list  #
            save_dict['t_labels'] = t_label_list
            save_dict['v_labels'] = v_label_list
            
            """model_architecture = [nn.Sequential(
                            model_vgg16,
                            Squeeze(),
                            nn.Linear(4096,11),
                            nn.Softmax(dim=0),
                        )]"""

            save_dict['model.state_dict'] = model.state_dict()# .to('cpu')
            #save_dict['model_architecture_untrained'] = model_architecture

            title = save_dict['Run']
            with open(f'{save_location}{title}.pkl', 'wb+') as f:
                pickle.dump(save_dict, f)
            
            print('improvment in metrics. model saved')

            #print('END Of SAve -----------------------------')
            #!nvidia-smi

        

        #if (epoch+1)%2==0:
            #train_log(t_loss, v_loss, epoch)
            #wandb.log({'train_accuracy_%': train_acc, 'epoch':epoch})
            #wandb.log({'val_accuracy_%': val_acc, 'epoch':epoch})
            
    model = best_model

    return model,save_dict



class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
    

    
def pipeline(config): 
    
    resnet = nn.Sequential(
            newmodel,
            Squeeze(),
            nn.Linear(2048,11), #2048
            nn.Softmax(dim=0),
        )
    
    resnet.to(device)
    model=resnet
    loss_list=[]
    model, save_dict = train_model(model,x_train, x_val, y_train, y_val, config) #train_dl, val_dl""

    return model, save_dict

"""with wandb.init(project=title, config=config):
    config = wandb.config
    model = resnet

    model, save_dict = train_model(model,x_train, x_val, y_train, y_val, config) #train_dl, val_dl""

return model, save_dict"""


model, save_dict = pipeline(config)

# confusion matrix


def plot_confusion(predictions:list, actual:list, title:str):
    predict_list = [int(t.argmax()) for t in predictions]
    actual = [int(l.argmax()) for l in actual]

    actual = np.array(actual)
    predict_list = np.array(predict_list)


    #FixedLocator locations (3), usually from a call to set_ticks, does not match the number of labels (11).
    print(f'\n     {title}')
    train_epoch_matrix = confusion_matrix(actual, predict_list, labels= [0,1,2,3,4,5,6,7,8,9,10])
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9,10])
    disp.plot()
    plt.show()


t_predict = save_dict['t_predict_list']
t_labels = save_dict['t_labels']

v_predict = save_dict['v_predict_list'] # WHY IS THERE NOTHING IN V OREDICT LIST!
v_labels = save_dict['v_labels']

plot_confusion(t_predict, t_labels, 'Train Confusion Matrix')
plot_confusion(v_predict, v_labels, 'Validation Confusion Matrix')