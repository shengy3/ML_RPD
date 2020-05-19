import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.Visualization import plot_residual
from lib.Fitting import fit_gaussian, fit_double_gaussian
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"savefig.bbox": 'tight'})
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPool2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow
from sklearn.model_selection import train_test_split

from lib.PerformanceEvaluator import get_RMS, load_data, get_event_in_range, case_list, get_mask
import tensorflow.keras as keras
from lib.RPD_CM_calculator import RPD_CM_calculator

from lib.Visualization import plot_residual
from lib.Fitting import fit_gaussian, fit_double_gaussian
%load_ext autoreload
%autoreload 2
#tensorflow.keras.backend.set_epsilon(1)


def evaluate_prediction(model, input_ary, truth, case_name, i, tag = ''):
    #predict the distribution
    predict = model.predict(input_ary)    
    #calculate the CoM from ML output (Recon_CM) and ground truth(True_CM)
    Recon_CM = RPD_CM_calculator(predict.reshape(-1,4,4), correction = False)
    #set up the position of each channel in x/y
    Recon_CM.X_pos_array = np.array([-15, -5, 5, 15])
    Recon_CM.Y_pos_array = np.array([15, 5, -5, -15])
    Recon_CM.calculate_CM()    
    #calculate the CoM from ML output (Recon_CM) and ground truth(True_CM)
    True_CM = RPD_CM_calculator(truth.reshape(-1,4,4) , correction = False)
    #set up the position of each channel in x/y
    True_CM.X_pos_array = np.array([-15, -5, 5, 15])
    True_CM.Y_pos_array = np.array([15, 5, -5, -15])
    True_CM.calculate_CM()
    #calculate the residual
    CM_residual = True_CM.CM - Recon_CM.CM
    
    #set up the range for histogram
    r = 10
    hist_range = [-r, r]
    hist_range2D = [hist_range, hist_range]
    output_folder = "./Output/fig/Energy_weight_training/"
    
    #plot residual
    single_gaussian_plt_para = {'fit_function':fit_gaussian,
    "init_para" :(10,1, 1),#init fit constant for the gaussian
    "n_bins": 200,#number of bin for the historgram in fit_range_def
    "fit_range_def": (-10, 10),#the range for gaussian fitting
    "range_def": (-10, 10),#the whole range of the histogram 
    "xlim": [-3, 3],#the range of the plot in x
    "density": True,#normalize histogram to density
    "output_path":output_folder + f"{tag}_CM_{case_name}_single_gaussian_residual_{i}.pdf"}#output figure
    
    ax, fig = plot_residual(CM_residual, **single_gaussian_plt_para)   
    plt.close('all')
    
    #plot 2d histogram 
    hist_range2D = [hist_range, hist_range]
    plt.hist2d(Recon_CM.CM[:,0], Recon_CM.CM[:,1], bins = 40, normed=True, range = hist_range2D)
    plt.xlim(hist_range)
    plt.ylim(hist_range)
    plt.title("CoM prediction")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Normalized number of events', rotation=270)
    plt.savefig(output_folder + f"{tag}_Recon_CM_{case_name}_2Dprediction_{i}.pdf")
    plt.close()
    
def get_data_set(bias, gpos, truth, normalization = False, flatten = False, pad = 1, test_size=0.3):
    
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

def train_model(start_case = 0):
    case_list = ["120GeV_neutron_uniplane_HAD",  "400GeV_neutron_uniplane_HAD", "1TeV_neutron_uniplane_HAD", "2.5TeV_neutron_uniplane_HAD"]
    
    for case in range(start_case, 4):
        #set up the parameter
        test_size = 0.3
        normalization = True
        flatten = True
        pad = 1
        n_epochs = 1
        
        #read in the data
        PATH = f"./Data/{case_list[case]}.pickle"
        df = pd.read_pickle(PATH)
        truth = df["Truth_44"].to_numpy()
        bias = df["RPD"].to_numpy()
        gpos = np.array([[x, y] for x, y in zip(df["gunPosX"].to_numpy(), df["gunPosY"].to_numpy())])


        loss_function_tag = 'BinaryX'

        tra_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth = get_data_set(bias,\
                                                                                    gpos,\
                                                                                    truth,\
                                                                                    normalization = True,\
                                                                                    flatten = True,\
                                                                                    pad = 1,
                                                                                    test_size = test_size)
        #generate the validation set (center 10x10)
        center_tra_bias, center_tra_gpos, center_truth = get_event_in_range(val_bias, val_gpos, val_truth)

        model = generate_model()
        
        model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=["kullback_leibler_divergence"], )
        #train for 10 epoch * 20 times
        for i in range(20):
            history = model.fit(tra_bias, tra_truth, epochs=n_epochs, batch_size=1000, validation_data = (val_bias, val_truth))     
            model.save(f'./Output/Model/Energy_reso_{loss_function_tag}_{case_list[case]}_{i}.h5')
            evaluate_prediction(model, center_tra_bias, center_truth, case_list[case], i, tag = loss_function_tag)

            
            
            
if __name__ == '__main__':
    train_model(start_case = 3)