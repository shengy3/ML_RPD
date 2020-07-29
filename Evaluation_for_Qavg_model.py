import tensorflow.keras as keras
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"savefig.bbox": 'tight'})
from lib.Visualization import plot_residual, get_residual_subplot
from lib.Fitting import fit_gaussian, fit_double_gaussian
from lib.Dataloader import get_training_and_validation


if __name__ == '__main__':

	_, _, _, _, v_ini_hit, v_signal, v_inic_q_avg, v_inic_q_std = get_training_and_validation(data_path)

	model = keras.models.load_model(f'./Output/Model/V4_BOX_Random_pos_fix_box_Q_vector150.h5')

	output = model.predict([v_ini_hit, v_signal])


	valid_Qavg  = output - v_inic_q_avg


	single_gaussian_plt_para = {'fit_function':fit_gaussian,
	"init_para" :(10,1, 1),
	"n_bins": 100,
	"range_def": (-3, 3),
	"fit_range_def": (-3, 3),
	"xlim": [-3, 3],
	"density": True,
	"output_path": output_folder + f"residual_.pdf"}

	plot_residual(valid_avg, **single_gaussian_plt_para)