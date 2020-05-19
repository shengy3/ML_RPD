import numpy as np


class RPD_CM_calculator():
    def __init__(self,feature_array, correction = True,  normalization = False):
        self.feature_array = feature_array
        self.correction = correction
        self.normalization = normalization
        self.event_photon = self.get_event_photon()
        self.corrected_RPD = []
        self.CM = []
        self.residual =[]
        #
        self.X_pos_array = np.array([0, -15, -5, 5, 15, 0])
        self.Y_pos_array = np.array([0, 15, 5, -5, -15, 0])
        #
        self.CM=[]
        self.process_array = []
        
    def get_event_photon(self):
        event_photon = []
        for event in self.feature_array:
            event_photon.append(np.sum(event))
        return np.array(event_photon)

    def substrct_row(self, feature):
        corrected_row = []
        for i in range(feature.shape[0]):
            if i == 0:
                corrected_row.append(feature[0])
            else:
                corrected_row.append(feature[i] - feature[i-1])
        corrected_row = np.array(corrected_row).clip(min=0).reshape((self.X_pos_array.shape[0],self.X_pos_array.shape[0]))
        if self.normalization is True:
            corrected_row = self.normalize_feature(corrected_row)
        return corrected_row
    
    def correct_RPD(self):
        print("Corrrect for the RPD")
        corrected_RPD = []
        for feature in self.feature_array:
            corrected_RPD.append(self.substrct_row(feature))
        self.corrected_RPD = np.array(corrected_RPD)
    
    def calculate_CM(self):
        print("Calculate center of mass")
        CM = []
        if self.correction is True:
            self.correct_RPD()
            process_array = self.corrected_RPD
        elif self.correction is False:
            process_array = self.feature_array
        #check for normalization
        if np.sum(process_array[0]) != 1.0:
            process_array = self.normalize_feature(process_array)
        for feature in process_array:
            #print(feature.shape)
            X = round(np.sum(feature * self.X_pos_array), 2)
            Y = round(np.sum(feature.T * self.Y_pos_array), 2)
            CM.append([X, Y])
        self.process_array = process_array
        self.CM = np.array(CM)
    
    def normalize_feature(self, process_array):
        process_array = process_array.reshape(-1, self.X_pos_array.shape[0], self.X_pos_array.shape[0])
        nol_array = np.zeros_like(process_array)
        for i, feature in enumerate(process_array):
            nol_array[i] = feature / np.sum(feature)
        return nol_array
    
    def get_residual(self):
        self.correct_RPD()
        self.calculate_CM()
        self.residual = self.analyzer.calculate_residual(self.CM, self.target_array)
        print("Done")