import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.models import Model
import tensorflow
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from lib.PerformanceEvaluator import get_event_in_range, get_data_set

%load_ext autoreload
%autoreload 2

def generate_model():
    
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (1,1),padding = 'Same',
                     activation ='relu', input_shape = (6,6,1)))
    model.add(Conv2D(filters = 16, kernel_size = (2,2),padding = 'Same',
                     activation ='relu'))
    model.add(Conv2D(filters = 16, kernel_size = (2,2),padding = 'Same',
                     activation ='relu'))

    model.add(Flatten())
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(64, activation = "relu"))

    model.add(Dense(32, activation = "relu"))
    model.add(Dense(20, activation = "relu"))

    model.add(Dense(16, activation = "sigmoid"))
    
    return model 

def train_model(**Training_para):
    case_list = ["120GeV_neutron_uniplane_HAD",  "400GeV_neutron_uniplane_HAD", "1TeV_neutron_uniplane_HAD", "2.5TeV_neutron_uniplane_HAD"]
    
    for case in range(Training_para['start_case'], 4):

        
        #read in the data
        loss_function_tag = 'BinaryX'

        tra_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth = get_data_set(case_list[case],\
                                                                                     normalization = Training_para['normalization'],\
                                                                                     flatten = Training_para['flatten'],\
                                                                                     pad = Training_para['pad'],
                                                                                    test_size = Training_para['test_size'] )
        #generate the validation set (center 10x10)
        center_tra_bias, center_tra_gpos, center_truth = get_event_in_range(val_bias, val_gpos, val_truth)

        model = generate_model()
        
        model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=["kullback_leibler_divergence"], )
        #train for 10 epoch * 20 times
        for i in range(Training_para['n_repeat']):
            history = model.fit(tra_bias, tra_truth, epochs=Training_para['n_epochs'], batch_size=1000, validation_data = (val_bias, val_truth))     
            model.save(f'./Output/Model/Energy_reso_{loss_function_tag}_{case_list[case]}_{i}.h5')

            
            
"""
case_list = ["120GeV_neutron_uniplane_HAD", 
             "400GeV_neutron_uniplane_HAD",
             "1TeV_neutron_uniplane_HAD",
             "2.5TeV_neutron_uniplane_HAD"]
             
Note: instaed using recall in the keras, here we use n_epochs to as one iteration, 
it is easier to implant the evaluation step after each evaluation.
The total training epochs = n_epochs * n_repeat
"""

Training_para ={'test_size' : 0.3, #the size in training/test split
        'normalization' : True, # noramlized the input and the output
        'flatten' : True, #flatten the output in put 1x16 array
        'pad' : 1, #add padding to the input 2D arraay
        'n_epochs' : 10, #number of epoch per interation for training 
        'n_repeat': 10, #
       'start_case': 3}# select the start case in case list to train. 
            
if __name__ == '__main__':
    train_model(**Training_para)
