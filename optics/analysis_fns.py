# Functions for loading in pickle files and for post train analysis

# imports

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
#import wget
import pandas as pd

import os
from torch.nn import functional
from zipfile import ZipFile
import cv2

import tensorflow as tf

import pickle
#import umap.umap_ as umap
import seaborn as sns
from numpy.linalg import norm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
#from boxsdk import OAuth2, Client


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        try:
            embedding_dict = pickle.load(f)
        except NameError:
            print('file path not recognised')
    return embedding_dict

#def split_by_label(embedding_dict, num_classes:int, label_name:str):
#    for i in range(num_classes):
#        embed+str(i) = embedding_dict.loc[embedding_dict[label_name]== str(i)]



# data from dict, split by classes, added to list for analysis
#
##
#



# learning curve

def learning_curve(t_loss_list, v_loss_list, epochs, save_location, run_name):
  sns.set()
  lab = "Learning Curve "+run_name
  plt.title(label=lab, fontsize =30)
  plt.plot(range(len(t_loss_list)), t_loss_list, label ='Training loss') #range(len(t_loss_list))
  plt.plot(range(len(v_loss_list)), v_loss_list, label='Validation loss') #range(len(v_loss_list))
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  #plt.yscale("log")
  plt.legend()
  plt.savefig(save_location+lab+'.png') #run_name
  plt.show()
  # save figs

def lil_learning_curve(loss_list, epochs, save_location, run_name):
  sns.set()
  lab = "Learning Curve "+run_name
  plt.title(label=lab, fontsize =30)
  plt.plot(range(len(loss_list)), loss_list, label ='Training loss') #range(len(t_loss_list))
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  #plt.yscale("log")
  plt.legend()
  plt.savefig(save_location+lab+'.png') #run_name
  plt.show()
  # save figs

# accuracy cuve
def accuracy_curve(t_accuracy_list, v_accuracy_list, epochs, save_location, run_name):
  sns.set()
  lab = "Accuracy Curve "+str(run_name)
  plt.title(label= lab, fontsize =30)
  plt.plot(range(epochs), t_accuracy_list, label ='Training accuracy') #range(len(t_accuracy_list))
  plt.plot(range(epochs), v_accuracy_list, label='Validation accuracy') #range(len(v_accuracy_list))
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.savefig(save_location+lab+'.png', format='png')
  plt.show()




def split_by_class(embedding_dict, num_class, class_name):
  class_list = []
  for i in range(num_class):
    embedding_l[i] = embedding_dict.loc[embedding_dict[str(class_name)] == i]
    class_list.append(embedding[l[i]])
  return class_list


def cosine_sim(a, b):
  c_dist = np.dot(a, b)/(norm(a)*norm(b))
  return c_dist

def cosine_distance(a, b):
  c_dist = 1- cosine_sim(a,b)
  return 
  
def cosine_mean_2loc(location1, location2):
    # compares similarity between two classes after training, classes must be pre split.
  if len(location1) >= len(location2):
    bigloc = location1
    littleloc = location2
  else:
     bigloc = location2
     littleloc = location1

  dist_list = []
  for i in range(len(littleloc)):
    sample1 = np.array(littleloc.iloc[i]['embedding'])
    for j in range(len(bigloc)):
      sample2 = np.array(bigloc.iloc[i]['embedding'])
      dist = cosine_distance(sample1, sample2)
      dist_list.append(dist)
  mean = sum(dist_list)/ len(dist_list)
  return mean

  def compare_classes(location_list):
    # applies cosine_mean_2loc on all classes. takes in a list of classes
    mean_cosine_list = []
    for idx, i in enumerate(location_list):
        for jdx, j in enumerate(location_list):
            a = cosine_mean_2loc(i, j)
            mean_cosine_list.append(a)
    return mean_cosine_list


def pickle_to_conf_matrix(mean_cosine_list, embedding_dict, labels, num_classes:int):
    sns.set()
    reshaped_cosine = np.array(mean_cosine_list)
    reshaped_cosine = reshaped_cosine.reshape(num_classes, num_classes)
    print('Cosine Distance Heat Map of '+num_classes+' Classes') # informal title
    cosine_matrix = sns.heatmap(reshaped_cosine)
    predict_list = [int(x) for x in embedding_dict['predictions']]
    labels = [int(x) for x in labels]
    labels = np.array(labels)
    predict_list = np.array(predict_list)
    epoch_matrix= confusion_matrix(labels, predict_list)
    label_list = []
    label_list = [list.append(int(i)) for i in range(num_classes)]
    disp = ConfusionMatrixDisplay(epoch_matrix, display_labels=label_list)
    disp.plot()
    plt.show()


def cluster_map(mean_cosine_list, num_classes):
    sns.set()
    sns.set_theme()
    reshaped_cosine = np.array(mean_cosine_list)
    reshape_cosine = reshaped_cosine.reshape(num_classes, num_classes)

    df = pd.DataFrame.from_dict(embedding_dict)
    df['embedding'] = df['embedding'].apply(np.array)
    df['location'] = pd.to_numeric(df['location'])
    df['predictions'] = df['predictions'].apply(np.array)

    netword_pal = sns.husl_palette(num_classes, s=.45)
    network_lut = dict(zip(df['location']).astype(str), network_pal)
    numeric_locs = df['location'][pd.to_numeric(df['location'])]
    network_colours = pd.Series(numeric_locs.astype(str)).map(network_lut)
    row_colours = netwoork_colours.to_numpy()

    g = sns.clustermap(reshaped_cosine, center =0, cmap="vlag",
    row_colors =row_colours, col_colors= network_colours.tolist(),
    dendrogram_ratio=(.1, .2),
    cbar_pos=(0.02, 0.32, 0.03, 0.3),
    linewidths=0.75,
    figsize=(12,13))

    g.ax_row_dendrogram.remove()


def plot_confusion(predictions:list, actual:list, title:str):
    sns.set()
    predict_list = [int(t) for t in predictions] #.argmax()
    actual = [int() for l in actual] #l.argmax()

    actual = np.array(actual)
    predict_list = np.array(predict_list)


    #FixedLocator locations (3), usually from a call to set_ticks, does not match the number of labels (11).
    print(f'\n     {title}')
    train_epoch_matrix = confusion_matrix(actual, predict_list, labels= [0,1,2,3,4,5,6,7,8,9,10])
    disp= ConfusionMatrixDisplay(train_epoch_matrix, display_labels=[0,1,2,3,4,5,6,7,8,9,10])
    disp.plot()
    plt.show()

