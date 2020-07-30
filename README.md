# Convolutional neural network for reaction plane reconstruction

This the deep learning model using convolutional neural network (CNN) to reconstruct the reaction plane for the nuclear 
collision events.


# Problem formulation:


<p align="center">
<img src="https://github.com/shengy3/ML_RPD/blob/master/images/Experiment%20setup.png" width="600" height="250">
</p>

When a high particle strikes a nuclide in a nuclear detector, the particle breaks the nuclide into fragements and create multiple 
charged particles along the incident direction of the incident particle. The shape of created charged particles, called shower, is a cone with the vertex 
at the interaction point. If we place a 2D array detector after the nuclear detector, we could a projection of the shower and be able to resonstruct the position of the incident particles. However, due to the limited space and the resource, the 2D array detector has overlapping area that leads to the distortion of the signal distribution. 

<p align="center">
<img src="https://github.com/shengy3/ML_RPD/blob/master/images/RPD_structure.png" width="324" height="250">
</p>

In addition, in the reality, there are maybe up to 40 particle incident in the same time. This makes the incident position of the particles very difficult to estimate. 

This project is to develop a neural network to reconstruct the original energy distribution and estimate the average of the incident particles, Q-vector. The model has two inputs: 1. CNN for the 2D array signal, 2. the number of incident particles and particles that create a shower. Right now, the model for energy reconstruction and Q-vector has the same model structure except for the output layer and the loss function. The visialization of the Q-vector model is:

<p align="center">
<img src="https://github.com/shengy3/ML_RPD/blob/master/images/ML_model_structure.png" align="center" width="150" height="350" >
</p>

# Training and evaluation

Please find the training data set in this link:https://uofi.box.com/s/0bmqesd2mnqmnknap08fso61cdwjjby9  
Then, put the downloaded file to the Data folder.  
You can train the model by 

    python Train_energy_resolution_model.py # for energy reconstruction model
    python Train_Qavg_model.py # for Q-vector model

The output model will be in *./Output/Model*.
For now, the inputs are a 2 dimension tuple and 6x6 array, 4x4 pixel with 1 padding. The output is the 1 x 16 array for energy reconstruction and 2 dimension tuples for Q-vector estimation.

You can performa the evaluation in *Evaluation_for_eng_resolu_model.ipynb* or by
    
    python Evaluation_for_Qavg_model.py
# Results:
The difference between predicted Q-vector and ground truth is fitted with a Gaussian function. The standard deviation of the fitted gaussian is known as spatial resolution. The spatial resolution of the neural network in x and y direction is 0.26 and 0.35 mm respectively. The resolution is **30 times** smaller than the pixel of the detector. 
<p align="center">
<img src="https://github.com/shengy3/ML_RPD/blob/master/images/residual_result.png" width="600" height="350">
</p>

# libray:
    
    -Visualization.py
    def plt_residual(residual, **plt_para):
        residual: it take array with dimension (events, x, y) and draw the histogram of each evetns
        plt_para: take the keywords for the plot
        
        example: 
        'fit_function':fit_gaussian,
        "init_para": initial fit constant for the gaussian. (10, 1, 1)
        "n_bins": number of bin for the historgram in fit_range_def, 200
        "fit_range_def": the range for gaussian fitting. (-10, 10)
        "range_def": the whole range of the histogram, (-10, 10)
        "xlim": the range of the plot in x, [-3, 3],
        "density": normalize histogram to density, True
        "output_path": output path for figure, " "

    -PerformanceEvaluator.py
    def get_data_set(case, normalization = False, flatten = False, pad = 1, test_size=0.3)
        """
        case: str, the case name to load
        normalization: bool, noramlize the photon arrival at the  PMT (bias) (4x4), and photon vertex (truth)(4x4) by its max in the event
        flatten: bool, flatten the photon vertex (4x4) into 16x1 array
        pad: int, add the padding to the bias
        test_size: float, the ratio of the test set in the whole data set

        return:
        bais: array in (events, width, height, channel(1)) dimension
        truth: dependent on the flatten flag. flatten is True (events, 16, 1), flatten is False: (events, 4, 4, 1)d imension
        gpos: array in (events, x, y) dimension

        example:
        train_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth = get_data_set(case_list[case])
    
    def get_event_in_range(bias, gpos, truth, lim = 10):
        """
        get the events based on the gpos within lim

        example:
        train_bias, tra_gpos, truth = get_event_in_range(train_bias, tra_gpos, tra_truth)
        """
    
    -RPD_CM_calculator.py
    class RPD_CM_calculator():
    def __init__(self,feature_array, correction = True,  normalization = False):
        """
        feature_array: np array, the array for calculate center of mass. The default array dimension (event, 6,6)
        correction: bool, whether do the row subtraction correction
        normalization: bool, whether normalize each events to its summation of signal over all the channel
        calculate_CM: calculat the CM for each events
        CM: the CoM result of the each event. (events, x, y)
        """
