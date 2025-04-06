import numpy as np
from sklearn.model_selection import train_test_split

def load_CW_Source(in_file,sec,byte):

    raw_trace=np.load(f'{in_file}trace_byte{byte}.npy')
    raw_label=np.load(f'{in_file}label_V_byte{byte}.npy')
    plaintext=np.load(f'{in_file}plaintext_byte{byte}.npy')


    profiling_traces = raw_trace[0:sec]
    profiling_label = raw_label[0:sec]
    plaintext_train = plaintext[0:sec]


    test_traces = raw_trace[sec:]
    test_label = raw_label[sec:]
    plaintext_test = plaintext[sec:]


    return profiling_traces, profiling_label, test_traces, test_label, plaintext_train, plaintext_test

def load_CW_Target_validation(in_file,sec,byte):

    raw_trace=np.load(f'{in_file}trace_byte{byte}.npy')
    raw_label=np.load(f'{in_file}label_V_byte{byte}.npy')
    plaintext=np.load(f'{in_file}plaintext_byte{byte}.npy')


    profiling_traces = raw_trace[0:sec]
    profiling_label = raw_label[0:sec]
    plaintext_train = plaintext[0:sec]


    test_traces = raw_trace[sec:]
    test_label = raw_label[sec:]
    plaintext_test = plaintext[sec:]


    return profiling_traces, profiling_label, test_traces, test_label, plaintext_train, plaintext_test



def load_CW_Target(in_file,byte):
    sec = 20000
    raw_trace = np.load(f'{in_file}trace_byte{byte}.npy')
    raw_label = np.load(f'{in_file}label_V_byte{byte}.npy')
    plaintext = np.load(f'{in_file}plaintext_byte{byte}.npy')

    return raw_trace, raw_label, plaintext

