# @Time     : Jan. 10, 2019 15:15
# @Author   : Veritas YIN
# @FileName : math_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import numpy as np
import pandas as pd
import math


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return ((x - mean) / std)*1


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return (x/1) * std + mean

# OLD METHODS
############################################
# NEW METHODS

def scale(x, p1, p2, scalation_type="robust_original"):
    if scalation_type == "z_score":
        return z_score(x, p1, p2)
    elif scalation_type == "robust":
        return robust_scale(x, p1, p2)
    elif scalation_type == "robust_original":
        return robust_scale_original(x, p1, p2)
    elif scalation_type == "log_scale":
        return log_scale(x, p1, p2)
    elif scalation_type == "none":
        return x
    else:
        raise ValueError("No accepted scalation type!!")
        
    
def descale(x, p1, p2, scalation_type="robust_original"):
    if scalation_type == "z_score":
        return z_inverse(x, p1, p2)
    elif scalation_type == "robust":
        return inverse_robust_scale(x, p1, p2)
    elif scalation_type == "robust_original":
        return inverse_robust_scale_original(x, p1, p2)
    elif scalation_type == "log_scale":
        return inverse_log_scale(x, p1, p2)
    elif scalation_type == "none":
        return x
    else:
        raise ValueError("No accepted scalation type!!.")

        
def get_stats(data, scalation_type):
    if scalation_type == "z_score" or scalation_type == "none":
        return {'mean': np.mean(data), 'std': np.std(data)}
    elif scalation_type == "robust":
        mean, std = get_column_data(data)
        return {'mean': mean, 'std': std}
    elif scalation_type == "robust_original":
        return {'mean': np.mean(data), 'std': np.percentile(data, 75) - np.percentile(data, 25)}
    elif scalation_type == "log_scale":
        return {'mean': np.mean(np.log1p(data)), 'std': np.std(np.log1p(data))}
    else:
        raise ValueError("No accepted scalation type!!.")



def get_column_data(data):
    aux_mean = np.zeros( (len(data[:, 0, 0, 0]), 1, len(data[0, 0, :, 0]), len(data[0, 0, 0, :])) )
    aux_iqr = np.zeros( (len(data[:, 0, 0, 0]), 1, len(data[0, 0, :, 0]), len(data[0, 0, 0, :])) )
    for c1 in range(len(data[:, 0, 0, 0])):
        for c3 in range(len(data[0, 0, :, 0])):
            for c4 in range(len(data[0, 0, 0, :])):
                aux_mean[c1, 0, c3, c4] = data[c1, :, c3, c4].mean()
                aux_iqr[c1, 0, c3, c4] = np.percentile(data[c1, :, c3, c4], 75) - np.percentile(data[c1, :, c3, c4], 25)
                if aux_iqr[c1, 0, c3, c4] == 0:
                        aux_iqr[c1, 0, c3, c4] = 1
                        
                       
    return aux_mean, aux_iqr


def robust_scale(data, mean, irq):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''
    # Aplicar la fórmula de normalización robusta
    scaled_data = np.zeros_like(data)
    for c1 in range(len(data[:, 0, 0, 0])):
        for c3 in range(len(data[0, 0, :, 0])):
            for c4 in range(len(data[0, 0, 0, :])):
                scaled_data[c1, :, c3, c4] = (data[c1, :, c3, c4] - mean[c1, 0, c3, c4]) / irq[c1, 0, c3, c4]

    
    return scaled_data


def inverse_robust_scale(scaled_data, mean, iqr):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized inversed array.
    '''
    # Realizar la inversión de la fórmula de normalización robusta
    data = np.zeros_like(scaled_data)
    for c1 in range(len(scaled_data[:, 0, 0, 0])):
        for c3 in range(len(scaled_data[0, 0, :, 0])):
            for c4 in range(len(scaled_data[0, 0, 0, :])):
                data[c1, :, c3, c4] = (scaled_data[c1, :, c3, c4] * irq[c1, 0, c3, c4]) + mean[c1, 0, c3, c4]

    
    
    return data


def robust_scale_original(x, mean, irq):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''
    scaled_data =  (x - mean) / irq
    
    return scaled_data


def inverse_robust_scale_original(x, mean, iqr):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized inversed array.
    '''
    inv_scaled_data = (x * iqr) + mean
    
    return inv_scaled_data


def log_scale(x, mean, std):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized array.
    '''
    normalized_data = np.log1p(x)
    
    return normalized_data


def inverse_log_scale(x, mean, std):
    '''
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param iqr: float, interquartile range.
    :return: np.ndarray, robust normalized inversed array.
    '''
    inv_normalized_data = np.expm1(x)
    
    return inv_normalized_data


# NEW METHODS
############################################
# OLD METHODS


def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    # Crear un índice booleano para los valores infinitos
    v_aux = pd.Series(v_[0, :, 0])
    vaux = pd.Series(v[0, :, 0])
    data = (abs((v_aux - vaux)/(vaux))).replace(np.inf, 0)
    
    return np.mean(data)


def MAPE_antiguo(v, v_):
    '''
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))

def RMSE(v, v_):
    '''
    Mean squared error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, RMSE averages on all elements of input.
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAE averages on all elements of input.
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth.
    :param y_: np.ndarray or int, prediction.
    :param x_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    dim = len(y_.shape)
    
    #print("Tamaño de la predicción:", dim)
    
    if dim == 3:
        # single_step case
        ##### ORIGINAL
        #v = inverse_robust_scale_original(y, x_stats['mean'], x_stats['std'])
        #v_ = inverse_robust_scale_original(y_, x_stats['mean'], x_stats['std'])
        
        #v = z_inverse(y, x_stats['mean'], x_stats['std'])
        #v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        
        ##### NEW! WE USE THE SCALING INSTEAD OF THE REAL ONE, FOR TESTING PURPOSES.
        v = y
        v_ = y_
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=-1)
