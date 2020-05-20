import numpy as np



case_list = ["120GeV_neutron_uniplane_HAD",\
            "400GeV_neutron_uniplane_HAD",\
            "1TeV_neutron_uniplane_HAD",\
            "2.5TeV_neutron_uniplane_HAD"]

def get_RMS(residual):
    """
    calcaulate the RMS of the input array(residual)
    example:
    print(get_RMS(residual))
    """
    RMS = lambda ary: np.sqrt(np.mean(ary**2))
    return RMS(residual[:, 0]),  RMS(residual[:, 1])


def get_data_set(case, normalization = False, flatten = False, pad = 1, test_size=0.3):
    
        """
        case: str, the case name to load
        normalization: bool, noramlize the photon arrival at the  PMT (bias) (4x4), and photon vertex (truth)(4x4)
        flatten: bool, flatten the photon vertex (4x4) into 16x1 array
        pad: int, add the padding to the bias
        test_size: float, the ratio of the test set in the whole data set
        
        example:
        train_bias, val_bias, tra_gpos, val_gpos, tra_truth, val_truth = load_data(case_list[case])
        """
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

def get_event_in_range(bias, gpos, truth, lim = 10):
    """
    get the events based on the gpos within lim
    
    example:
    train_bias, tra_gpos, truth = get_event_in_range(train_bias, tra_gpos, tra_truth)
    """

    mask = get_mask(gpos, -lim, lim, -lim, lim)
    bias = bias[mask]
    gpos = gpos[mask]
    truth = truth[mask]
    return bias, gpos, truth

def get_mask(ary, xmin, xmax, ymin, ymax):
    """
    get the union in the ary by given range
    
    x1 = gpos[:,0] > -lim
    x2 =  gpos[:,0] < lim
    y1 = gpos[:,1] > - lim
    y2 = gpos[:,1] < lim
    
    mask = x1&x2&y1&y2
    """
    x1 = ary[:,0] > xmin
    x2 =  ary[:,0] < xmax
    y1 = ary[:,1] > ymin
    y2 = ary[:,1] < ymax
    mask = x1&x2&y1&y2
    return mask
