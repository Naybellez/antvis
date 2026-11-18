# fucntions for compatibility with WANB


import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from loop_fns import loop
from functions import get_data



#                    OPTIMISERS

def build_optimizer(network, optimizer, learning_rate, weight_decay=0):
    """
    Returns torch optimiser
    Optimiser type is specified by 'optimizer' arguemnt, string of ' SGD' or 'adam'
    """
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                                     lr=learning_rate, weight_decay=weight_decay)
    return optimizer



def set_optimizer(optim, model, learning_rate):
    """
    Returns list of different optimizers - differentiated via optimizer type (SGD or adam) or by weight decay value
    Used for hyperparameter sweeps
    """
    optim_list=[]
    if optim =='Adam':
        optimizer1 = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer2 = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)
        optim_list.append(optimizer1)
        optim_list.append(optimizer2)
    elif optim == 'SGD':
        optimizer3 = torch.optim.SGD(model.parameters(), lr=learning_rate)
        optim_list.append(optimizer3)
    return optim_list


def set_lossfn(lf):
    """
    Returns loss function
    loss function type is controlled via 'lf' argument: string of 'MSE' or 'CrossEntropy'
    """
    if lf =='MSE':
        loss_fn = nn.MSELoss()
    elif lf == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    return loss_fn


#           PIPLINE FUNCTIONS
#from loop_fns import print_gpu_mem
                                # HP Sweep
def hp_sweep(config, col_dict,save_dict, device,seed,model, loss_fn, optimizer, scheduler, best_acc=0, data=None):
    """
    Hyperparameter sweep 
    following the sweep outlined on the Wandb tutorial site.
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_data(file_path= config.image_path, seed=seed)

    e_count = 0
    t_loss_list = []
    v_loss_list = []#
    t_predict_list = []
    v_predict_list = []#
    t_accuracy_list = []#
    v_accuracy_list = []
    t_label_list = []#
    v_label_list = []
    #prepro = ImageProcessor(device)
    
    
    for epoch in tqdm(range(config.epochs)):

        t_correct =0
        v_correct =0

        
        model.train()
        print('training...')

      
        t_loss, t_predictions, t_num_correct,t_labels, model, optimizer = loop(model, x_train[:10], y_train[:10], epoch, loss_fn, device, col_dict, num_classes=config.num_classes, pad_size=col_dict['padding'],optimizer=optimizer, scheduler=scheduler)

        t_accuracy = (t_num_correct /len(x_train))*100
        t_accuracy_list.append(t_accuracy)

        t_label_list.append(t_labels)
        t_predict_list.append(t_predictions)
        t_loss_list.append(t_loss)

        model.eval()
        print('validating...') 
        v_loss, v_predictions, v_num_correct, v_labels= loop(model, x_val[:10], y_val[:10], epoch, loss_fn, device,col_dict, num_classes= config.num_classes, train=False) 
        v_accuracy= (v_num_correct / len(x_val))*100

        v_label_list.append(v_labels)
        v_predict_list.append(v_predictions)
        v_loss_list.append(v_loss)
 
        
        
        # logging
        wandb.log({'train_loss': t_loss, 'epoch':epoch})
        wandb.log({'val_loss': v_loss, 'epoch':epoch})
        wandb.log({'train_accuracy_%': t_accuracy, 'epoch':epoch})
        wandb.log({'val_accuracy_%': v_accuracy, 'epoch':epoch})

        save_dict['Current_Epoch'] = config['epochs']
        save_dict['training_samples'] = len(x_train)# should this be the whole list for future graphs...?
        save_dict['validation_samples'] = len(x_val)
        save_dict['t_loss_list'] = t_loss_list #[c.to('cpu') for c in t_loss_list]
        save_dict['t_predict_list'] = [[c.to('cpu') for c in k]for k in t_predict_list] #[[c.to('cpu') for c in k]for k in t_predict_list]  # [c.to('cpu') for c in t_predict_list] 
        save_dict['t_accuracy_list'] = t_accuracy_list #
        save_dict['v_loss_list'] = v_loss_list #[c.to('cpu') for c in v_loss_list]
        save_dict['v_predict_list'] = [[c.to('cpu') for c in k]for k in v_predict_list]#[[c.to('cpu') for c in k]for k in v_predict_list] # [c.to('cpu') for c in v_predict_list]
        save_dict['v_accuracy_list'] = v_accuracy_list #
        save_dict['t_labels'] = [[c.to('cpu') for c in k]for k in t_label_list]
        save_dict['v_labels'] = [[c.to('cpu') for c in k] for k in v_label_list]

        e_count +=1 # epoch count

    return model, save_dict, x_test, y_test
