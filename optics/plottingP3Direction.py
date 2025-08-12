import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns
from plotting import check_save_path

def plot_confusion(predictions:list, actual:list, title:str, run_name:str,save_location =None):
    #this wasn't designed to be given a list of batches
    print(len(predictions), len(actual))
    save_location = check_save_path(save_location)
    sns.set()
    #print(predictions)
    
    if type(predictions[0]) != int and type(predictions[0]) != list:
        predict_list = [int(t.argmax()) for t in predictions] ##
        predict_list = [int(t.numpy()) for t in predictions]
    else:
        predict_list = predictions
        
    if type(actual[0])!= int:
        actual = [int(l.argmax()) for l in actual]
         
    actual = np.array(actual)
    predict_list = np.array(predict_list)

    
    font1 = {'family':'serif','color':'darkblue','size':16}
    font2 = {'family':'serif','color':'darkblue','size':15}
    
    label = np.zeros(360, dtype='float32')

    train_epoch_matrix = confusion_matrix(actual, predict_list, labels= label)
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels= label)
    
    disp.plot(cmap='plasma')
    plt.title(run_name+'\n'+title, font1) #label="Accuracy Curve \n"+title, font1)
    plt.xlabel('Predicted Label', font2)
    plt.ylabel('Target Label', font2)
    if save_location != None:
        plt.savefig(save_location+'/'+'Conf_mtrx'+title+run_name+'.png', format='png')
    else:
        print("Save Location Not Specified!")
    plt.show()