import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import seaborn as sns
from plotting import check_save_path

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
    print(np.unique(actual))
    predict_list = [int(np.round(i/10)) for i in predict_list]
    print(np.unique(predict_list))
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


def plot_predictions(preds, targets, num_samples=5):

    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    plt.figure(figsize=(10, num_samples * 2))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(targets[i], label="Target", color='black', linewidth=2)
        plt.plot(preds[i], label="Prediction", color='red', linestyle='--')
        plt.title(f"Sample {i} | Target Peak : {(targets[i].argmax()).item()} | Pred Peak : {(preds[i].argmax()).item()}")
        plt.legend()
    plt.tight_layout()
    plt.show()
