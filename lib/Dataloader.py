import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from lib.Dataloader import get_training_and_validation


def process_RPD_signal(ary, normalization = False, flatten = False, pad = 1):
    if normalization:
            ary = np.array([i.reshape(4,4,1)/np.max(i) for i in ary])
    else:
        ary = np.array([i.reshape(4,4,1) for i in ary])
    if flatten:
            ary = np.array([i.reshape(16) for i in ary])
    if pad:
        ary = np.pad(ary[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')
    return ary

def get_input(ary):
    init_hit = ary[0]
    signal = process_RPD_signal(ary[1])
    inic_q_avg = ary[2]
    inic_q_std = ary[3]
    return init_hit, signal, inic_q_avg, inic_q_std

def convert_np_array(ary):
    return np.array([i for i in ary])

def process_dataset(ary):
    n_event = ary[:,0]
    hit_signal = ary[:, 1]
    inic_q_avg = ary[:, 7]
    init_pos = ary[:,-2]
    Bbox_size = ary[:,-1]
   
    out = []
    for item in [n_event, hit_signal, inic_q_avg, init_pos, Bbox_size]:
        out.append(convert_np_array(item))
        
    return out

def load_dataset(data_path):
	out = None
	for file in os.listdir(data_path):
	    if not file.endswith(".npy"):
	        continue
	    if out is None:
	        out = np.load( f"{data_path}{file}", allow_pickle = True)
	    else:
	        start = time.time()
	        tmp = np.load(f"{data_path}{file}",allow_pickle=True)
	        out = np.concatenate((out, tmp), axis = 0)
	return out

def get_training_and_validation(data_path):
	dataset = load_dataset(data_path)
	train, valid = train_test_split(dataset, test_size=0.3, random_state = 42)
	process_train = process_dataset(train)
	process_valid = process_dataset(valid)


	t_ini_hit, t_signal, t_inic_q_avg, t_inic_q_std = get_input(process_train)
	v_ini_hit, v_signal, v_inic_q_avg, v_inic_q_std = get_input(process_valid)

	return t_ini_hit, t_signal, t_inic_q_avg, t_inic_q_std, v_ini_hit, v_signal, v_inic_q_avg, v_inic_q_std