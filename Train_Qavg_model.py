from lib.Dataloader import get_training_and_validation
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense,  Flatten, Conv2D,  
from tensorflow.keras.models import Model




def get_model():
	# define two sets of inputs
	n_neutron = Input(shape=(2,), name = 'interact_neutron') # (40, ta_n)
	# the first branch operates on the first input
	x = Dense(2, activation="relu")(n_neutron)
	x = Dense(4, activation="relu")(x)
	x = Model(inputs=n_neutron, outputs=x)


	# the second branch opreates on the second input
	bias = Input(shape=(6,6,1), name = 'biased_RPD')
	y = Conv2D(filters = 16, kernel_size = (1,1),\
	           padding = 'Same', activation ='relu')(bias)
	y = Conv2D(filters = 16, kernel_size = (2,2),\
	           padding = 'Same', activation ='relu')(y)
	y = Flatten()(y)

	y = Dense(32, activation = "relu")(y)
	y = Dense(64, activation = "relu")(y)
	y = Dense(64, activation = "relu")(y)
	y = Dense(20, activation = "relu")(y)
	y = Model(inputs=bias, outputs=y)

	# combine the output of the two branches
	combined = keras.layers.concatenate([x.output, y.output])
	combined = Dense(20, activation="relu", name = 'combined')(combined)
	combined = Dense(20, activation="relu")(combined)


	# apply a FC layer and then a regression prediction on the
	# combined outputs
	Q_avg = Dense(20, activation="relu")(combined)
	Q_avg = Dense(10, activation="relu")(Q_avg)
	Q_avg = Dense(2, activation="linear", name = 'Q_avg')(Q_avg)

	model = Model(inputs=[x.input, y.input], outputs=[Q_avg])
	return model

if __name__ == '__main__':
	data_path = "./Data/V4_MC2_varying_incident_XYstd_combined/"

	t_ini_hit, t_signal, t_inic_q_avg, t_inic_q_std, v_ini_hit, v_signal, v_inic_q_avg, v_inic_q_std = get_training_and_validation(data_path)

	model = get_model()
	model.compile(optimizer='adam',loss=['mae'])

	model.fit(
    {"interact_neutron": t_ini_hit, 'biased_RPD': t_signal},
    #{"Q_avg": t_inic_q_avg, "Q_std": box_size},
    { "Q_avg": t_inic_q_avg},
    
    epochs= 150,
    batch_size=10000,
     validation_data = ({"interact_neutron": v_ini_hit, 'biased_RPD': v_signal},\
                        {"Q_avg": v_q_avg})
	)

	model.save(f'./Output/Model/V4_BOX_Random_pos_fix_box_Q_vector150.h5')