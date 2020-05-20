Please find the training data set in this link:https://uofi.box.com/s/0bmqesd2mnqmnknap08fso61cdwjjby9  
Then, put the downloaded file to the Data folder.  
You can train the model by python Energy_resolution_model.py  
The output model will be in ./Output/Model  
For now, the input is 6x6 array, 4x4 channels with 1 padding. The output is the 1 x 16 array.


lib:
    
    -Visualization.py
    ```python
    def plt_residual(residual, **plt_para):
    ```
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
    
    get_event_in_range(bias, gpos, truth, lim = 10):
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
