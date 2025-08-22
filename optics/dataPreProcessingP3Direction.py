import os
import numpy as np
from sklearn.model_selection import train_test_split


def import_imagedata(file_path, locations:list): 
    # ensure locations is a list
    if locations is None:
        locations = [1,2,3,4,5,6,7,8,9,10,11]
    elif isinstance(locations, str):
        locations = [locations]
    if any(not isinstance(x, int) for x in locations):
        raise TypeError("locations list should be list of int(s) 0-11")
    if not isinstance(locations, list):
        raise TypeError("import_imagedata expects locations as list!")
        
    # import image data from dir
    images = []
    labels = []
    print(file_path)

    for file in os.listdir(file_path):
        if file[0:4] == 'IDSW':
            if int(file[4:7]) in locations:
                j = file_path+file
                i=int(file[5:7]) -1
                i = str(i)
                labels.append(i)
                images.append(j)
    label_arr =np.array(labels)
    image_arr = np.array(images)
    return image_arr, label_arr

def get_data(random_seed, file_path:str, locations:list):
    if file_path is None:
        raise TypeError("get_data requires file path argument")
    elif not isinstance(file_path, str):
        raise TypeError(f"get_data requires file path as string not {type(file_path)}")
    #print(file_path)
    img_len = len(os.listdir(file_path))
    x, y = import_imagedata(file_path, locations)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, train_size=0.7,
                                     random_state=random_seed, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.3, train_size=0.7,
                                     random_state=random_seed, shuffle=True)
    return x_train, y_train, x_val, y_val, x_test, y_test