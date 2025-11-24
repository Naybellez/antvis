import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns
from plotting import check_save_path
import os
import re

def checkSaveName(saveloc, savename):
    print("checkSaveName Start")
    name, ext = os.path.splitext(savename)
    match = re.search(r"_(\d+)$", name)
    if match:
        base = name[:match.start()]
        i = int(match.group(1))
        print("checkSaveName match")
    else:
        base =  name
        i = 0
        print("checkSaveName else")
    new_name = savename
    print("checkSaveName while loop starting")
    if os.path.exists(os.path.join(saveloc,savename)):
        i +=1
        new_name = f"{base}_{i}"
    print("checkSaveName while loop end")
    return new_name


def plot_confusion(predictions:list, actual:list, title:str, run_name:str,save_location =None):
    #this wasn't designed to be given a list of batches
    #print(len(predictions), len(actual))
    save_location = check_save_path(save_location)
    sns.set()
    #print(predictions)
    
    if type(predictions[0]) != int and type(predictions[0]) != list:
        predict_list = [int(t.argmax()) for t in predictions] ##
        predict_list = [int(t.numpy()) for t in predictions]
        print(predict_list[0], type(predict_list[0]))
    else:
        predict_list = predictions
        
    if type(actual[0])!= int:
        actual = [int(l.argmax()) for l in actual]

    actual = [int(np.round(i/10)) for i in actual]
    print("plot_confusion ACTUAL",np.unique(actual))
    predict_list = [int(np.round(i/10)) for i in predict_list]
    print("plot_confusion PREDICTION",np.unique(predict_list))
    actual = np.array(actual)
    predict_list = np.array(predict_list)

    
    font1 = {'family':'serif','color':'darkblue','size':16}
    font2 = {'family':'serif','color':'darkblue','size':15}
    
    #label = np.zeros(36, dtype='float32') # 360
    label = np.arange(0, 36, 1)
    #print(f"confmatrx labels  {type(label)}   {label.shape}   {label}")

    train_epoch_matrix = confusion_matrix(actual, predict_list, labels= label)
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels= label)
    #disp= ConfusionMatrixDisplay.from_estimator()
    
    disp.plot(cmap='plasma')
    plt.title(run_name+'\n'+title, font1) #label="Accuracy Curve \n"+title, font1)
    plt.xlabel('Predicted Label', font2)
    plt.ylabel('Target Label', font2)
    if save_location != None:
        plt.savefig(save_location+'/'+'Conf_mtrx'+title+run_name+'.png', format='png')
    else:
        print("Save Location Not Specified!")
    plt.show()


def plot_predictions(preds, targets, peakdists, num_samples=5, runname=""):
    print("plot_predictions ACTIVATED")
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    #print(type(peakdists), peakdists)
    print("plot_predictions CPU'D")
    plt.figure(figsize=(10, num_samples * 2))
    print("plot_predictions PLOTTING START")
    for i in range(num_samples):
        
        plt.subplot(num_samples, 1, i+1)
        plt.plot(targets[i], label="Target", color='black', linewidth=2)
        plt.plot(preds[i], label="Prediction", color='red', linestyle='--')
        plt.title(f"Sample {i} | Target Peak : {(targets[i].argmax()).item()} | Pred Peak : {(preds[i].argmax()).item()} | PeakDist : {peakdists[i]}") #peakdists[i]
        if i < num_samples :
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
        plt.legend()
        print("plot_predictions PLOTTING END") 
        plt.tight_layout()
        savename = checkSaveName( "/its/home/nn268/antvis/antvis/optics/res_big_loop_saves/models/p3/NEWLABEL/","PlotPreds_"+runname) # saveloc, savename)
        plt.savefig(savename+".jpg")
        print("plot_predictions SAVED")
        plt.show()
