import tensorflow.keras as keras
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
plt.rcParams.update({"savefig.bbox": 'tight'})
from lib.Visualization import plot_residual, get_residual_subplot
from lib.Fitting import fit_gaussian, fit_double_gaussian
from lib.Dataloader import get_training_and_validation

def blur_neutron(n_hit):
    blur = []

    for i in range(n_hit.shape[0]):
	blur.append(np.random.normal(n_hit[i], n_hit[i] * 0.171702))
    return np.array(blur)

def get_unit_vector(arr):
    output = []
    norm =  np.linalg.norm(arr, axis = 1)
    for i in range(arr.shape[0]):
        output.append([arr[i][0] / norm[i], arr[i][1] / norm[i]])
    return np.array(output)

def process_signal(ary, normalization = False, flatten = False, pad = 1):
    if normalization:
            ary = np.array([i.reshape(4,4,1)/np.max(i) for i in ary])
    else:
        ary = np.array([i.reshape(4,4,1) for i in ary])
    if flatten:
            ary = np.array([i.reshape(16) for i in ary])
    if pad:
        ary = np.pad(ary[:, :, :, :], ((0, 0), (pad, pad), (pad, pad), (0,0)), 'constant')
    return ary


def average_vector(QA, QB):
    NormA = np.linalg.norm(QA, axis = 1)
    NormB = np.linalg.norm(QB, axis = 1)
    NA = np.array([QA[:,0 ] / NormA, QA[:,1 ] / NormA])
    NB = np.array([QB[:,0 ] / NormB, QB[:,1 ] / NormB])
    flip_B = -NB

    #AVG_v = (NA + NB) / 2
    #AVG_v = AVG_v.reshape(-1, 2)
    #avg = np.arctan2(AVG_v[:,1], AVG_v[:,0])
    avgx = (NA[0] + flip_B[0]) / 2     
    avgy = (NA[1] + flip_B[1]) / 2     
    avg = np.arctan2(avgy, avgx)    
    return NA, NB, avgx, avgy, avg

if __name__ == '__main__':
	uproot_installed = False

	outA = np.load("./Data/ToyV1_Fermi_2.7TeV_Merge_122420/validA.npy", allow_pickle = True)
	outB = np.load("./Data/ToyV1_Fermi_2.7TeV_Merge_122420/validB.npy", allow_pickle = True)

	A_signal = outA[:,8:]
	B_signal = outB[:,8:]

	A_inic_q_avg = outA[:, 0:2]
	B_inic_q_avg = outB[:, 0:2]


	A_hit = outA[:, 7]
	B_hit = outB[:, 7]


	Apsi_gen = np.arctan2(A_inic_q_avg[:,1],A_inic_q_avg[:,0])
	Apsi_true = outA[:,5].astype(float)


	Bpsi_gen = np.arctan2(B_inic_q_avg[:,1],B_inic_q_avg[:,0])
	Bpsi_true = outB[:,5].astype(float)


	A_signal = process_signal(A_signal)
	B_signal = process_signal(B_signal)


	#A_hit = blur_neutron(A_hit)
	#B_hit = blur_neutron(B_hit)

	model = keras.models.load_model(f'./Output/Model/ToyV1_Fermi_Merge_122420_unit_vector.h5', compile = False) 

	QA = model.predict([A_hit.astype(float), A_signal.astype(float)])
	QB = model.predict([B_hit.astype(float), B_signal.astype(float)])


        

	dQA  = A_inic_q_avg - QA
	dQB  = B_inic_q_avg - QB

	Adx = dQA[:,0]
	Ady = dQA[:,1]

	Bdx = dQB[:,0]
	Bdy = dQB[:,1]

	Apsi_rec = np.arctan2(QA[:,1],QA[:,0])
	Apsi_gen = np.arctan2(A_inic_q_avg[:,1],A_inic_q_avg[:,0])
	AR_rec = np.sqrt(QA[:,0] ** 2 + QA[:, 1] ** 2)


	Bpsi_rec = np.arctan2(QB[:,1],QB[:,0])
	Bpsi_gen = np.arctan2(B_inic_q_avg[:,1],B_inic_q_avg[:,0])
	BR_rec = np.sqrt(QB[:,0] ** 2 + QB[:, 1] ** 2)

	#A_inic_q_avg = get_unit_vector(A_inic_q_avg)
	#B_inic_q_avg = get_unit_vector(B_inic_q_avg)


	NA, NB, avg_recon_x, avg_recon_y, avg_recon_angle = average_vector(QA, QB)

	NA_gen, NB_gen, avg_gen_x, avg_gen_y, avg_gen_angle = average_vector(A_inic_q_avg, B_inic_q_avg)

	AX = A_inic_q_avg[:,0].astype(float)
	AY = A_inic_q_avg[:, 1].astype(float)
	AR = np.sqrt(AX**2 + AY**2)

	BX = B_inic_q_avg[:,0].astype(float)
	BY = B_inic_q_avg[:, 1].astype(float)
	BR = np.sqrt(BX**2 + BY**2)
	
	
	TreeA = {"n_incident_neutron": A_hit,
        "X_gen":AX,
         "Y_gen": AY,
         "R_gen": AR,
         "Qx_rec": QA[:,0],
         "Qy_rec": QA[:,1],
         "dX": Adx,
         "dY": Ady,
         "psi_true":Apsi_true,
         "psi_rec":Apsi_rec,
         "psi_gen": Apsi_gen,
         "R_rec": AR_rec}

	TreeB = {"n_incident_neutron": B_hit,
		 "X_gen":BX,
		 "Y_gen": BY,
		 "R_gen": BR,
		 "Qx_rec": QB[:,0],
		 "Qy_rec": QB[:,1],
		 "dX": Bdx,
		 "dY": Bdy,
		 "psi_true":Bpsi_true,
		 "psi_rec":Bpsi_rec,
		 "psi_gen": Bpsi_gen,
		 "R_rec": BR_rec}

	Tree_arms = {"NormAx":NA[0],
		     "NormAy":NA[1],
		     "NormBx": -NB[0],
		     "NormBy": -NB[1],
		     "Average_RP_vector_X": avg_recon_x,
		     "Average_RP_vector_Y": avg_recon_y,
		     "Average_RP_angle": avg_recon_angle
		    }
	
	
	if uproot_installed:
		from ROOT import TFile, TTree
		from array import array
		import uproot
		
		def fill_array(arr):
		    output = []
		    tree = arr.item()
		    for branch in tree.keys():
			print(branch)
			tmp = array( 'd' )
			for i in tree[branch]:
			    tmp.append(i)
			output.append(tmp)

		    return output
	
		A = fill_array(TreeA)
		B = fill_array(TreeB)
		arms = fill_array(Tree_arms)
		
		A = fill_array(TA)
		B = fill_array(TB)
		arms = fill_array(Tarms)
		
		f = TFile(f'{pt}pt.root', 'recreate')
		
		armA = TTree('ARM A', 'tree')
		armB = TTree('ARM B', 'tree')
		Avg = TTree('Avg RP', 'tree')

		tmpA = [array('d',[0]) for _ in range(len(A))]
		tmpB = [array('d',[0]) for _ in range(len(B))]
		tmparm = [array('d',[0]) for _ in range(len(arms))]
		
		
		armA.Branch('n_incident_neutron',tmpA[0], 'n_incident_neutron/D')
		armA.Branch('X_gen',tmpA[1], 'X_gen/D')
		armA.Branch('Y_gen', tmpA[2], 'Y_gen/D')
		armA.Branch('R_gen', tmpA[3], 'R_gen/D')
		armA.Branch('Qx_rec', tmpA[4], 'Qx_rec/D')
		armA.Branch('Qy_rec', tmpA[5], 'Qy_rec/D')
		armA.Branch('dX', tmpA[6], 'dX/D')
		armA.Branch('dY', tmpA[7], 'dY/D')
		armA.Branch('psi_true', tmpA[8], 'psi_true/D')
		armA.Branch('psi_rec', tmpA[9], 'psi_rec/D')
		armA.Branch('psi_gen', tmpA[10], 'psi_gen/D')
		armA.Branch('R_rec', tmpA[11], 'R_rec/D')


		armB.Branch('n_incident_neutron',tmpB[0], 'n_incident_neutron/D')
		armB.Branch('X_gen', tmpB[1], 'X_gen/D')
		armB.Branch('Y_gen', tmpB[2], 'Y_gen/D')
		armB.Branch('R_gen', tmpB[3], 'R_gen/D')
		armB.Branch('Qx_rec', tmpB[4], 'Qx_rec/D')
		armB.Branch('Qy_rec', tmpB[5], 'Qy_rec/D')
		armB.Branch('dX', tmpB[6], 'dX/D')
		armB.Branch('dY', tmpB[7], 'dY/D')
		armB.Branch('psi_true', tmpB[8], 'psi_true/D')
		armB.Branch('psi_rec', tmpB[9], 'psi_rec/D')
		armB.Branch('psi_gen',tmpB[10], 'psi_gen/D')
		armB.Branch('R_rec',tmpB[11], 'R_rec/D')



		Avg.Branch('NormAx', tmparm[0], 'NormAx/D')
		Avg.Branch('NormAy', tmparm[1], 'NormAy/D')
		Avg.Branch('NormBx', tmparm[2], 'NormBx/D')
		Avg.Branch('NormBy', tmparm[3], 'NormBy/D')
		Avg.Branch('Average_RP_vector_X', tmparm[4], 'Average_RP_vector_X/D')
		Avg.Branch('Average_RP_vector_Y', tmparm[5], 'Average_RP_vector_Y/D')
		Avg.Branch('Average_RP_angle', tmparm[6], 'Average_RP_angle/D')



		nentries = len(TA.item()['n_incident_neutron'])
		for i in range(nentries):
		    for j in range(len(tmpA)):
			tmpA[j][0] = A[j][i]

		    for j in range(len(tmpB)):
			tmpB[j][0] = B[j][i]

		    for j in range(len(arms)):
			tmparm[j][0] = arms[j][i]

		    armA.Fill()
		    armB.Fill()
		    Avg.Fill()
		f.Write()
		f.Close()
