# file for  plots

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns


def check_save_path(save_location):
    if save_location is not None:
        if save_location[-1]!= '/':
            save_location = save_location+'/'
    return save_location

def plot_confusion(predictions:list, actual:list, title:str, run_name:str,save_location =None):
    #this wasn't designed to be given a list of batches
    print(len(predictions), len(actual))
    save_location = check_save_path(save_location)
    sns.set()
    #print(predictions)
    if type(predictions[0]) != int and type(predictions[0]) != list:
        predict_list = [int(t.argmax()) for t in predictions] ##
        #print("2 p ",predict_list[:10], type(predict_list))
        #print(predictions)
        predict_list = [int(t.numpy()) for t in predictions]
    else:
        predict_list = predictions

    #print("1  a ", actual[:10], type(actual))
    if type(actual[0])!= int:
        actual = [int(l.argmax()) for l in actual]
    #print("2  a ", actual[:10], type(actual))
    actual = np.array(actual)
    #print("3  a ", actual[:10], type(actual))
    predict_list = np.array(predict_list)
    #print(len(predict_list), len(actual))
    
    font1 = {'family':'serif','color':'darkblue','size':16}
    font2 = {'family':'serif','color':'darkblue','size':15}

    train_epoch_matrix = confusion_matrix(actual, predict_list, labels= [0,1,2,3,4,5,6,7,8,9,10])
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9,10])
    disp.plot(cmap='plasma')
    plt.title(run_name+'\n'+title, font1) #label="Accuracy Curve \n"+title, font1)
    plt.xlabel('Predicted Label', font2)
    plt.ylabel('Target Label', font2)
    if save_location != None:
        plt.savefig(save_location+'/'+'Conf_mtrx'+title+run_name+'.png', format='png')
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

def learning_curve(t_loss, v_loss, save_location,run_name:str):
    save_location = check_save_path(save_location)
    print(f"SAVING TO ... {save_location}")
    lab = "Learning Curve "+run_name
    font1 = {'family':'serif','color':'darkblue','size':16}
    font2 = {'family':'serif','color':'darkblue','size':15}
    
    plt.plot(range(len(t_loss)), t_loss, label ='Training loss')
    plt.plot(range(len(v_loss)), v_loss, label='Validation loss')
    plt.title(run_name+"\n Learning Curve ", font1)
    plt.xlabel('Epochs', font2)
    plt.ylabel('Loss', font2)
    #plt.yscale("log")
    plt.legend()
    if save_location != None:
        plt.savefig(save_location+lab+'.png') #run_name
    else:
        print("Save Location Not Specified!")
    plt.show()

def accuracy_curve(t_accuracy_list, v_accuracy_list,save_location,run_name:str):
    save_location = check_save_path(save_location)
    lab = "Accuracy Curve"+run_name
    font1 = {'family':'serif','color':'darkblue','size':16}
    font2 = {'family':'serif','color':'darkblue','size':15}

    plt.title(run_name+"\n Accuracy Curve", font1)
    plt.plot(range(len(t_accuracy_list)), t_accuracy_list, label ='Training accuracy')
    plt.plot(range(len(v_accuracy_list)), v_accuracy_list, label='Validation accuracy')
    plt.xlabel('Epochs', font2)
    plt.ylabel('Accuracy', font2)
    plt.legend()
    if save_location != None:
        plt.savefig(save_location+'/'+lab+'.png', format='png')
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


def create_polar_plot(df_coordinates, max_len):
    # takes in a pandas df
    # max_len is the max length of all df to be plotted. used to visually see the time taken- time from frames
    # convert data from df series to np.array for plotting
    _x = np.array(df_coordinates['x'])
    _y =  np.array(df_coordinates['y'])

    # Convert to polar coordinates
    r = np.sqrt(_x**2 + _y**2)  # radius
    theta = np.arctan2(_y, _x)  # angle in radians
    
    # Create the polar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    # Plot the points
    ax.scatter(theta, r, alpha=0.5, c=df_coordinates['frame'], vmin=0, vmax=max_len, cmap="gist_rainbow") #nipy_spectral

    ax.set_thetagrids(np.arange(0, 360, 30),
                      labels=['180°', '150°', '120°', '90°', '60°', '30°',
                             '0°', '330°', '300°', '270°', '240°', '210°'])
    # Customize the plot
    ax.set_title('Coordinate Points in Polar Space')
    ax.grid(True)
    
    return fig