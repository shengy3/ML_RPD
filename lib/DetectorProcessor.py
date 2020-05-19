import numpy as np 
from .SignalProcessor import filter_signal


class Processor():
    def __init__(self, detector):
        self.detector = detector
        self.num_event = self.detector.channels[0].events.shape[0]
        
    def get_event_max(self):
        print("get_event_max")
        num_event = self.num_event
        event_max = np.zeros(num_event)
        for i, ch in enumerate(self.detector.channels): 
            event_max = np.zeros(num_event)
            for j, ev in enumerate(ch.events):
                # need to make the avg range flexible 
                mean = np.mean(ev[50:200])
                ev = filter_signal(ev) - mean
                event_max[j] = np.max(abs(ev))
            ch.signal_max = event_max
            
class RPDProcessor(Processor):
    def __init__(self, detector):
        self.detector = detector
        self.num_event = self.detector.channels[0].events.shape[0]

    def recon_RPD(self):
        self.get_event_max()
        print("Recon RPD")
        num_event = self.num_event
        recon_detector = np.zeros((num_event, 4, 4))
        for event in range(num_event):
            for ch in self.detector.channels:
                row = ch.channel_info['mapping_row'] - 1
                col = ch.channel_info['mapping_column'] - 1
                recon_detector[event, row, col] = ch.signal_max[event]
        #due to the mapping is start from the right
        recon_detector = np.array([np.flipud(np.fliplr(i)) for i in recon_detector])
        self.recon_detector = recon_detector
        
class ZDCProcessor(Processor):
    def __init__(self, detector):
        self.detector = detector
        self.num_event = self.detector.channels[0].events.shape[0]