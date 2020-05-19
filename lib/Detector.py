import matplotlib.pyplot as plt
import numpy as np
import uproot
from .ConfigHandler import ConfigHandler


class BasicFeature():
    def __init__(self):
        self.channel_info = {'name': '',
                             'mapping_row': 0,
                             'delay': 0, 
                             'mapping_row': 0,
                             'mapping_column': 0,
                             'offset' : 0,
                             'HV': 0,
                             'is_on': False,
                             'Vop' :0.0
                            }
    def get_channel_info(self):
        return self.channel_info

class Channel(BasicFeature):
    def __init__(self, file, detector_channel, configure):
        #parameter for reading file
        super(Channel, self).__init__()
        self.file = file
        self.detector_channel = detector_channel
        self.events = 0
        self.configure = configure
        self.read_file(file, detector_channel)
        self.signal_max = []
        
    def read_file(self, file, detector_channel):
        #print (f"reading {detector_channel} in {file}")
        tree = uproot.open(file)["tree"]
        #print ("reading channel:", detector_channel)
        #print("loading channel from tree")
        df = tree.pandas.df([detector_channel])
        #print (df.index.names)
        #print ("unpacking dataframe")
        events = df.unstack(level=-1).values
        
        #print ("shape of events container", events.shape)
        #print ("data type of events container", type(events))
        self.events = events
        self.set_channel_parameter()
        #print("done")

    def set_channel_parameter(self):
        channel_idx = self.configure['name'] == self.detector_channel
        config = self.configure[channel_idx]
        for para in self.channel_info.keys():
            self.channel_info[para] = config[para].to_numpy()[0]

class Detector():
    def __init__(self, file, run_number, detecter_type):

        self.file = file
        self.config = ConfigHandler()
        self.detector_info = self.config.get_run_info(run_number, detecter_type)
        self.active_channels = self.detector_info[self.detector_info['is_on'] == True]['name'].tolist()
        self.channels = []
        self.construct_channels()
    
    def construct_channels(self):
        print("Reading active channels: ", self.active_channels)
        for channel in self.active_channels:
            self.channels.append(Channel(self.file, channel, self.detector_info))
        self.channels = np.array(self.channels)
        print("Done")
    
    def show_processed_result(self):
        pass
