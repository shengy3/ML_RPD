Please find the training data set in this link:https://uofi.box.com/s/0bmqesd2mnqmnknap08fso61cdwjjby9  
Then, put the downloaded file to the Data folder.  
You can train the model by python Energy_resolution_model.py  
The output model will be in ./Output/Model  
For now, the input is 6x6 array, 4x4 channels with 1 padding. The output is the 1 x 16 array.


lib:
    Visualization.py
    plt_residual(residual, **plt_para):
    
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

