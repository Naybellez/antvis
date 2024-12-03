# file type managment functions
import os
import csv
import sys
import json
import collections
import numpy as np
import torch 

#     #    SAVING TO FILE TYPE FUNCTIONS. #    #    #    TAKES A DICTIONARY   #    #   # 
def save2csv_nest_dict(nested_dict, file_name, save_location:str):
    # flattern nested dictionary
    flatterend_dict = {}
    for k,v in nested_dict.items():
        if isinstance(v, dict):
            for nested_key, nested_val in v.items():
                flatterend_dict[f"{k}_{nested_key}"] = nested_val
        else:
            flatterend_dict[k] =v
    columns = list(flatterend_dict.keys())
    
    with open(save_location+str(file_name)+'.csv', "a+", newline="") as f:
        # using dictwriter
        writer = csv.DictWriter(f, fieldnames=columns)
        # using writeheader function
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(flatterend_dict)
        f.close()

def check_obj4np(obj):
    # check dictionary values for json and csv
    if isinstance(obj, dict):
        return {key: check_obj4np(value) for key, value in obj.items()}
    if isinstance(obj,list):
        return [check_obj4np(item) for item in obj]
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj
    
def save2josn_nested_dict(nested_dict, file_name, save_location:str):
    # save to json
    nested_dict = check_obj4np(nested_dict)
    json_obj = json.dumps(nested_dict, indent=4)
    with open(save_location+str(file_name)+'.json', 'a+') as f:
        f.write(json_obj)
        f.close()

def save2csv(nested_dict, file_name, save_location:str):
    nested_dict = check_obj4np(nested_dict)
    columns = list(nested_dict.keys())
    path = os.path.join(save_location, file_name +".csv")
    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            # using dictwriter
            # using writeheader function
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(nested_dict)
            f.close()
    except IOError as e:
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError:
              print("could not convert to string")
    except:
              print("unexpected error: ", sys.exc_info()[0])
 
def save2json(nested_dict, file_name, save_location:str):
    nested_dict = check_obj4np(nested_dict)
    json_obj = json.dumps(nested_dict, indent=4)
    path = os.path.join(save_location, file_name+".json")
    with open(path, 'w') as f:
        f.write(json_obj)

        
    
#    #   READ IN  FROM FILE TYPE  FUNCTIONS  #   #  #    #   READ IN  FROM FILE TYPE  FUNCTIONS  #   #
def read_in_json(file_path, file_name):
    path = os.path.join(file_path, 'file_name')
    try:
        with open(path, 'r') as f:
            dj = json.load(f, object_pairs_hook= collections.OrderedDict) #obj, 
    except Exception as e:
        print("Error decoding Json")
        print(e)



# Helpful printing functions. Could probably be deleated
def print_run_header(learning_rate, optim, loss_fn):
    print('\n')
    print('LR: ', learning_rate)
    print('optimiser ', optim)
    print('loss fn: ', loss_fn)

def print_run_type(run_type: str):
    print('                  ----------------------')
    print(f' \n                  {run_type}... \n')
    print('                  ----------------------')



def print_top_results(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch):
    print('Top results from hyperparameter sweep:')
    print()
    print(best_optim, best_lossfn, best_lr, best_valaccuracy, best_epoch)