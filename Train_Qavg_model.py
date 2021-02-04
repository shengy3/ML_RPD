from lib.Dataloader import get_training_and_validation
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense,  Flatten, Conv2D,  
from tensorflow.keras.models import Model




def get_model():
	# define two sets of inputs
	n_neutron = Input(shape=(1,), name = 'interact_neutron') # (40, ta_n)
	# the first branch operates on the first input
	x = Dense(4, activation="relu")(n_neutron)
	x = Dense(4, activation="relu")(x)
	x = Model(inputs=n_neutron, outputs=x)


	# the second branch opreates on the second input
	bias = Input(shape=(6,6,1), name = 'biased_RPD')
	y = BatchNormalization()(bias)
	y = Conv2D(filters = 16, kernel_size = (1,1),\
		   padding = 'Same', activation ='relu')(y)
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
	Q_avg = Dense(2, activation="tanh", name = 'Q_avg')(Q_avg)

	model = Model(inputs=[x.input, y.input], outputs=[Q_avg])

	return model

def get_unit_vector(arr):
	output = []
	norm =  np.linalg.norm(arr, axis = 1)
	for i in range(arr.shape[0]):
	output.append([arr[i][0] / norm[i], arr[i][1] / norm[i]])
	return np.array(output)



def process_signal(ary, normalization = False, flatten = False, padding = 1):
	if normalization:
		ary = np.array([i.reshape(4,4,1)/np.max(i) for i in ary])
	else:
		ary = np.array([i.reshape(4,4,1) for i in ary])
	if flatten:
		ary = np.array([i.reshape(16) for i in ary])
	if padding:
		ary = np.pad(ary[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')
	return ary


def get_dataset(folder = "./Data/reduced_tree_tmp/", side = 'A'):
    output = []
    start = time.time()
    for i in range(100, 1000100, 100):
	if i % 100000 == 0:
	    print("event", i, "time", time.time() - start)
	    start = time.time()
	output.append(pd.read_pickle(folder + f"{side}_RPD_photon{i}.pickle"))
    return pd.concat(output, ignore_index = True).set_index('Event number').astype(float)

if __name__ == '__main__':
	
	test_size = 0.2



	A = get_dataset(side = 'A',folder = "./Data/ToyV1_Fermi_2.7TeV_Merge_122420/")
	B = get_dataset(side = 'B', folder = "./Data/ToyV1_Fermi_2.7TeV_Merge_122420/")

	A = A.drop_duplicates()

	B = B.drop_duplicates()


	trainA, validA, trainB, validB = train_test_split(A, B, test_size=test_size, random_state = 42)
	
	print("Save validation set")
	np.save("./Data/ToyV1_Fermi_2.7TeV_Merge_122420/validA.npy", validA)
	np.save("./Data/ToyV1_Fermi_2.7TeV_Merge_122420/validB.npy", validB)

	train = trainA.append(trainB).to_numpy()
	valid = validA.append(validB).to_numpy()


	train_signal = train[:,8:]
	valid_signal = valid[:,8:]

	t_inic_q_avg = train[:, 0:2]
	v_inic_q_avg = valid[:, 0:2]

	t_hit = train[:, 7]
	v_hit = valid[:, 7]



	t_unit_vector = get_unit_vector(t_inic_q_avg)
	v_unit_vector = get_unit_vector(v_inic_q_avg)

	t_signal = process_signal(train_signal)
	v_signal = process_signal(valid_signal)
		model = get_model()
		model.compile(optimizer='adam',loss=['mae'])

		model.fit(
	    {"interact_neutron": t_ini_hit, 'biased_RPD': t_signal},
	    { "Q_avg": t_inic_q_avg},

	    epochs= 150,
	    batch_size=10000,
	     validation_data = ({"interact_neutron": v_ini_hit, 'biased_RPD': v_signal},\
				{"Q_avg": v_q_avg})
		)

	model.save(f'./Output/Model/Fermi_model.h5')
