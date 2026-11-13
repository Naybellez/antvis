# Functions for loading in pickle files and for post train analysis

# imports
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from numpy.linalg import norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_pickle(file_path):
    """
    Function that handles opening and loading pickled dictionary
    """
    with open(file_path, 'rb') as f:
        try:
            embedding_dict = pickle.load(f)
        except NameError:
            print('file path not recognised')
    return embedding_dict


def cosine_sim(a, b):
  """
  Cosine similarity function
  """
  c_dist = np.dot(a, b)/(norm(a)*norm(b))
  return c_dist

def cosine_distance(a, b):
  """
  Calculates cosine distance
  """
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
    """
    Creates cosine similarity confusion matrix
    """
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
