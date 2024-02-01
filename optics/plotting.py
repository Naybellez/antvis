# file for  plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns

def plot_confusion(predictions:list, actual:list, title:str, save_location =None):
    sns.set()
    predict_list = [int(t.argmax()) for t in predictions] ##
    predict_list = [int(t.numpy()) for t in predictions]
    actual = [int(l) for l in actual]


    actual = np.array(actual)
    predict_list = np.array(predict_list)

    font1 = {'family':'serif','color':'darkblue','size':20}
    font2 = {'family':'serif','color':'darkblue','size':15}

    train_epoch_matrix = confusion_matrix(actual, predict_list, labels= [0,1,2,3,4,5,6,7,8,9,10])
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9,10])
    disp.plot(cmap='plasma')
    plt.title(title, font1)
    plt.xlabel('Predicted Label', font2)
    plt.ylabel('Target Label', font2)
    if save_location != None:
        plt.savefig(save_location+'Conf_mtrx'+title+run_name+'.png', format='png')
    else:
        print("Save Location Not Specified!")
    plt.show()

def metrics(label, prediction): #TypeError: Singleton array tensor(3) cannot be considered a valid collection.

    label= np.array(label.cpu())

    predictions_np = prediction.cpu().detach().numpy()
    #y_pred' parameter of f1_score must be an array-like or a sparse matrix. Got 7 instead.
    predicted_classes = np.argmax(predictions_np, axis=0)
    #print('metrics Label:   ', label)
    #print('metrics prediction   ', predicted_classes)
    #avg_f1_score = f1_score(label, predictions_np, average='macro')
    acc = accuracy_score(label, predicted_classes)
    
    return acc

def learning_curve(t_loss, v_loss, save_location, title=''):
    lab = "Learning Curve"+title
    font1 = {'family':'serif','color':'darkblue','size':20}
    font2 = {'family':'serif','color':'darkblue','size':15}
    
    plt.plot(range(len(t_loss_list)), t_loss_list, label ='Training loss')
    plt.plot(range(len(v_loss_list)), v_loss_list, label='Validation loss')
    plt.title(label="Learning Curve"+title, font1)
    plt.x_label('Epochs', font2)
    plt.ylabel('Loss', font2)
    #plt.yscale("log")
    plt.legend()
    if save_location != None:
        plt.savefig(save_location+'/'+lab+'.png') #run_name
    else:
        print("Save Location Not Specified!")
    plt.show()

def accuracy_curve(v_accuracy_list, t_accuracy_list, save_location, title=''):
    lab = "Accuracy Curve"+title
    font1 = {'family':'serif','color':'darkblue','size':20}
    font2 = {'family':'serif','color':'darkblue','size':15}

    plt.title(label=lab, font1)
    plt.plot(range(len(t_accuracy_list)), t_accuracy_list, label ='Training accuracy')
    plt.plot(range(len(v_accuracy_list)), v_accuracy_list, label='Validation accuracy')
    plt.xlabel('Epochs', font2)
    plt.ylabel('Accuracy', font2)
    plt.legend()
    if save_location != None:
        plt.savefig(save_location+lab+'.png', format='png')
    else:
        print("Save Location Not Specified!")
    plt.show()

def open_pickle(save_location, file_name):
    with open(save_location+file_name, 'rb') as f:
        save_dict = pickle.load(f)
    return save_dict



# mean cosin similarity

# split to individual columns (? - check if best way)

def plot_grad_flow(model):

    #### This function, used specifically for the visualisation of gradient flow, was found in the following stackoverflow thread: https://stackoverflow.com/questions/70394788/pytorch-adam-optimizer-dramatically-cuts-gradient-flow-compared-to-sgd
    model = model.to('cpu')
    named_parameters = model.named_parameters()
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        #print(n,p)
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            #print('gradddssss', p.grad)
            #print('p: ', p)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1) #  top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    model = model.to('gpu')