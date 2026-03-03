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
        print("check pred type: ",predict_list[0], type(predict_list[0]))
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
    print(f"actual  {actual}")
    print(f"preds  {predict_list}")

    
    font1 = {'family':'serif','color':'darkblue','size':14}
    font2 = {'family':'serif','color':'darkblue','size':12}
    
    #label = np.zeros(36, dtype='float32') # 360
    label = np.arange(0, 36, 1)
    disp_labels = np.arange(0, 36, 1)
    #print(f"confmatrx labels  {type(label)}   {label.shape}   {label}")

    train_epoch_matrix = confusion_matrix(actual, predict_list, labels = label)
    disp = ConfusionMatrixDisplay(train_epoch_matrix, display_labels = disp_labels)# label)
    #disp= ConfusionMatrixDisplay.from_estimator()
    print(f"plot_conf  len label {len(label)}   len disp_labels {len(disp_labels)}")
    
    disp.plot(cmap='plasma')
    plt.title(run_name+'\n'+title, font1) #label="Accuracy Curve \n"+title, font1)
    plt.xlabel('Predicted Label', font2)
    plt.ylabel('Target Label', font2)
    if save_location != None:
        plt.savefig(save_location+'/'+'Conf_mtrx'+title+run_name+'.png', format='png', bbox_inches="tight",)
    else:
        print("Save Location Not Specified!")
    plt.show()


def plot_predictions(preds, targets, peakdists, num_samples=5, runname="", save_loc =""):
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    num_samples = min(num_samples, len(preds))
    print(num_samples)
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, sharex=True,  figsize=(6.4, num_samples*2)) # *2
    print(len(axes))
    for i in range(num_samples): # range(num_samples):
        axes[i].plot(targets[i], label="Label", color='grey', linewidth=2) #black
        axes[i].plot(preds[i], label="Pred", color='red', linestyle='--')
        axes[i].set_title(f"Sample {i} | Target Peak : {targets[i].argmax().item()} | Pred Peak : {preds[i].argmax().item()} | PeakDist : {peakdists[i]}") 
        #if i < num_samples :
        if i != num_samples-1:
            axes[i].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)
    axes[0].legend(loc="right")
        
    #plt.legend()#loc='upper left')
    savename = checkSaveName(save_loc, f"PlotPreds_"+runname) # saveloc, savename)
    plt.savefig(save_loc+savename+".jpg", dpi=100, bbox_inches="tight")#, )
    #print("plot_predictions SAVED")
    plt.show()
