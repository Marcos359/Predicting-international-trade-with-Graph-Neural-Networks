# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import scale, get_stats

import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


    
# OLD METHODS
############################################
# NEW METHODS

def seq_gen_interpolation(len_seq, data_seq, offset, n_frame, n_route, partern_slot = 3, C_0=1):
    años_testing = 1 * partern_slot
    años_training  = n_frame - años_testing
    
    #años_training  = 4 * partern_slot + 1
    
    tmp_seq = np.zeros((len_seq, años_training + años_testing, n_route, C_0))
    print(tmp_seq.shape)
    
    for i in range(0, len_seq*partern_slot, partern_slot):
        sta = i + offset * partern_slot
        end = sta + años_training + años_testing
        print(sta, end)
        #print(tmp_seq[int(i/partern_slot), :, :, :].shape)
        #print(data_seq[sta:end, :].shape)
        
        tmp_seq[int(i/partern_slot), :, :, :] = np.reshape(data_seq[sta:end, :], [años_training+años_testing, n_route, C_0])
        #print(tmp_seq[int(i/3), 0, :, :])
        #print(tmp_seq[int(i/3), -1, :, :])
        
    return tmp_seq

def seq_gen_simple(len_seq, data_seq, offset, n_frame, n_route, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    # Cambiar al original para encadernar muchos paises!
    
    tmp_seq = np.zeros((len_seq, n_frame, n_route, C_0))
    for i in range(len_seq):
        sta = offset + i
        end = sta + n_frame
        print(sta, end)
        tmp_seq[i, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
        #print(tmp_seq[i, :, :, :])
    return tmp_seq


        
def data_gen_simple(file_path, data_config, n_route, n_frame=21, interpolation = False):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test, normalisation = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=0).values
        ##### ORIGINAL
        ##### data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')
    
    # seq_gen_simple(numero años, -, año inicial, -, -)
    if not interpolation:
        seq_train = seq_gen_simple(n_train, data_seq, 0, n_frame, n_route)
        seq_val = seq_gen_simple(n_val, data_seq, n_train, n_frame, n_route)
        seq_test = seq_gen_simple(n_test, data_seq, n_train + n_val, n_frame, n_route)
        # seq_test = seq_gen_simple(n_test, data_seq, n_train + n_val, n_frame, n_route)
    elif interpolation:
        seq_train = seq_gen_interpolation(n_train, data_seq, 0, n_frame, n_route, 3)
        seq_val = seq_gen_interpolation(n_val, data_seq, n_train, n_frame, n_route, 3)
        seq_test = seq_gen_interpolation(n_test, data_seq, n_train + n_val, n_frame, n_route, 3)
        # seq_test = seq_gen_simple(n_test, data_seq, n_train + n_val, n_frame, n_route)
    
    seq_val = seq_test
    
    
    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = get_stats(seq_train, normalisation)

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = scale(seq_train, x_stats['mean'], x_stats['std'], normalisation)
    x_val = scale(seq_val, x_stats['mean'], x_stats['std'], normalisation)
    x_test = scale(seq_test, x_stats['mean'], x_stats['std'], normalisation)

        
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset

# NEW METHODS
############################################
# OLD METHODS

def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    # n_slot es cuantos datos podemos sacar por cada dia
    n_slot = day_slot - n_frame + 1
    
    # Los datos originales se toman cada 5 min, por lo que como mucho un dia puede ser 288 ejemplos de datos!
    # Luego, nosotros vamos a querer coger 12 ejemplos para reaizar las predicciones, y 9 para ver si son correctas (creo)
    # Por lo que necesitamos 21 ejemplos por cada set de entrenamiento, por lo que cada dia solo nos
    # aporta 268 (el valor de n_slot) sets de entrenamiento y realmente len_seq es cuantos dias quiero sacar para cada secuencia.

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            #print(sta, end)
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
    return tmp_seq

def data_gen(file_path, data_config, n_route, n_frame=21):
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=0).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
