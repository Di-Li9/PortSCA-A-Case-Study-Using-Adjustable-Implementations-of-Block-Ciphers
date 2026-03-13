import numpy as np
from sklearn.model_selection import train_test_split

def load_CW_Source(in_file,sec):

    raw_trace = np.load(f'{in_file}traces.npy')
    raw_label = np.load(f'{in_file}label_V.npy')
    Cipertext= np.load(f'{in_file}Cipertext.npy')


    profiling_traces = raw_trace[0:sec]
    profiling_label = raw_label[0:sec]
    Cipertext_train = Cipertext[0:sec]
    # MDS_train = MDS[0:sec]

    test_traces = raw_trace[sec:20000]
    test_label = raw_label[sec:20000]
    Cipertext_test = Cipertext[sec:20000]
    

    return profiling_traces, profiling_label, test_traces, test_label, Cipertext_train, Cipertext_test# MDS_train, MDS_test




def load_CW_Target(in_file):
    raw_trace = np.load(f'{in_file}traces.npy')
    raw_label = np.load(f'{in_file}label_V.npy')
    Cipertext = np.load(f'{in_file}Cipertext.npy')
    print(raw_trace.shape,raw_label.shape,Cipertext.shape)

    return raw_trace, raw_label, Cipertext