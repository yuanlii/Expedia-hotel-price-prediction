import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

class seq2seq():
    def __init__(self):
        self.data = pd.DataFrame()
        # subset of data divided by week
        self.sequence_list = []
        # number of weeks cover by data
        self.weeks = 0
        self.extra_day = 0
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        
        return self.data
    
    def get_sequence_data(self):
        '''process data into seven days as a sequence'''
        # TODO: predict next 7 days' price based on previous 7 days
        self.weeks += len(self.data)//7
        self.extra_day += len(self.data) % 7
        i = 1
        while 1 <= i <= self.weeks:
            subset = self.data[(i-1)*7:i*7]
            self.sequence_list.append(subset)
            i+=1
        return self.sequence_list

    def implement_seq2seq(self):
        

    
            
            