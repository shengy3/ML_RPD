import numpy as np



case_list = ["120GeV_neutron_uniplane_HAD",\
            "400GeV_neutron_uniplane_HAD",\
            "1TeV_neutron_uniplane_HAD",\
            "2.5TeV_neutron_uniplane_HAD"]

def get_RMS(residual):
    """
    print(get_RMS(residual))
    """
    RMS = lambda ary: np.sqrt(np.mean(ary**2))
    return RMS(residual[:, 0]),  RMS(residual[:, 1])


def load_data(case_name, pad = 1):
    """
    train_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth = load_data(case_list[case])
    """
    data_set_output = "./Output/Train_val_set/"
    train_bias =np.load(data_set_output+f"{case_name}_train_bias.npy", allow_pickle=True)
    val_bias = np.load(data_set_output+f"{case_name}_val_bias.npy" , allow_pickle=True)
    tra_gpos = np.load(data_set_output+f"{case_name}_tra_gpos.npy" , allow_pickle=True)
    val_gpos = np.load(data_set_output+f"{case_name}_val_gpos.npy" , allow_pickle=True)
    tra_truth = np.load(data_set_output+f"{case_name}_tra_truth.npy" , allow_pickle=True)
    val_truth = np.load(data_set_output+f"{case_name}_val_truth.npy" , allow_pickle=True)

    return train_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth

def get_event_in_range(bias, gpos, truth, lim = 10):
    """
    train_bias, tra_gpos, truth = get_event_in_range(train_bias, tra_gpos, tra_truth)
    """
    """
    x1 = gpos[:,0] > -lim
    x2 =  gpos[:,0] < lim
    y1 = gpos[:,1] > - lim
    y2 = gpos[:,1] < lim
    mask = x1&x2&y1&y2
    """
    mask = get_mask(gpos, -lim, lim, -lim, lim)
    bias = bias[mask]
    gpos = gpos[mask]
    truth = truth[mask]
    return bias, gpos, truth

def get_mask(ary, xmin, xmax, ymin, ymax):
    x1 = ary[:,0] > xmin
    x2 =  ary[:,0] < xmax
    y1 = ary[:,1] > ymin
    y2 = ary[:,1] < ymax
    mask = x1&x2&y1&y2
    return mask