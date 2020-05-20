import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.models import Model
import tensorflow
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from lib.PerformanceEvaluator import get_event_in_range

%load_ext autoreload
%autoreload 2
#tensorflow.keras.backend.set_epsilon(1)


def get_data_set(case, normalization = False, flatten = False, pad = 1, test_size=0.3):
        PATH = f"./Data/{case}.pickle"
        df = pd.read_pickle(PATH)
        truth = df["Truth_44"].to_numpy()
        bias = df["RPD"].to_numpy()
        gpos = np.array([[x, y] for x, y in zip(df["gunPosX"].to_numpy(), df["gunPosY"].to_numpy())])
        
        train_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth = train_test_split(bias, gpos, truth, \
                                                                        test_size=test_size, random_state = 42)

        
        if normalization:
            tra_bias = np.array([i.reshape(4,4,1)/np.max(i) for i in train_bias])
            val_bias = np.array([i.reshape(4,4,1)/np.max(i) for i in val_bias])
            tra_truth = np.array([i.reshape(4,4,1)/np.max(i) for i in tra_truth])
            val_truth = np.array([i.reshape(4,4,1)/np.max(i) for i in val_truth])

        else:
            tra_bias = np.array([i.reshape(4,4,1) for i in train_bias])
            val_bias = np.array([i.reshape(4,4,1) for i in val_bias])
            tra_truth = np.array([i.reshape(4,4,1) for i in tra_truth])
            val_truth = np.array([i.reshape(4,4,1) for i in val_truth])

        
        if flatten:
                tra_truth = np.array([i.reshape(16) for i in tra_truth])
                val_truth = np.array([i.reshape(16) for i in val_truth])
        if pad:
            tra_bias = np.pad(tra_bias[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')
            val_bias = np.pad(val_bias[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')
        
        return tra_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth
    
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
