# @Time     : Jan. 10, 2019 17:52
# @Author   : Veritas YIN
# @FileName : tester.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation, MAE, descale
from os.path import join as pjoin

# Arreglo para que tensorflow funcione en v1, ya que parece que todo se ha hecho en esa versión.
import tensorflow
from tensorflow.compat import v1 as tf

import numpy as np
import time

import pandas as pd

def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []

        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            
            ############### Control ###############
            # 50: It is the size of the batch.
            # 50 (batch_size) of all possible "blocks" (which for train would be 9112), are the sequences that we pass to predict.
            # 228: The value of the radars.
            # 13: The number of history, what we use to test and not to predict at first.
            # The first 13 (n_his) are used to predict, but then move the window to include the predictions themselves and then the predictions of the predictions.
            ############### Control ###############
            
            # ¿Que hace esto?
            if isinstance(pred, list):
                pred = np.array(pred[0])
                
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
        
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    
    return pred_array[step_idx], pred_array.shape[1]



def multi_pred_TFM(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, x_stats, normalisation, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []

        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})

            if isinstance(pred, list):
                pred = np.array(pred[0])
                
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
        
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)


    
        ############################# SAVING EACH YEARS PREDICTIONS #####################################################
    for x in range(step_idx[0]+1):
        print(x, " and ", step_idx[0]+1)

        print(seq.shape)
        x_guardar = pd.DataFrame(descale(seq[:, n_his+x, :, 0], x_stats['mean'], x_stats['std'], normalisation))
        x_guardarZ = pd.DataFrame(seq[:, -3+x, :, 0]) # 6-3 --n_his es de 6, el menos 3 es por el espacio!
        
        y_guardar = pd.DataFrame(descale(pred_array[x, :, :, 0], x_stats['mean'], x_stats['std'], normalisation))
        y_guardarZ = pd.DataFrame(pred_array[x, :, :, 0])
        aux_guardar = pd.DataFrame([x_stats['mean'], x_stats['std']])
        
        # 24+x = Año!
        x_guardar.to_csv(f'X_guardado_{24+x}.csv', index=False)
        x_guardarZ.to_csv(f'X_guardado_{24+x}Z.csv', index=False)
        y_guardar.to_csv(f'Y_guardado_{24+x}.csv', index=False)
        y_guardarZ.to_csv(f'Y_guardado_{24+x}Z.csv', index=False)
        aux_guardar.to_csv(f'Aux.csv', index=False)
    
    return pred_array[step_idx], pred_array.shape[1]


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')
    
    # "multi_pred" parece que realiza la predicción de una forma rara para superar los limites de tf.
    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    # "evaluation" parece que evalua la predicción, por lo que y_val es el valor predecido con x_val, el segmento de validación.
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)
    
    
    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    # update the metric on test set, if model's performance got improved on the validation.
    if sum(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        min_val = evl_pred
    return min_va_val, min_val


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, normalisation, load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

         ############################# SAVING AND CONTROL #####################################################
        # The evaluation has been rendered useless.
        # Now to evaluate we compare the results to ARIMA in Jupiter inside "Trials.ipynb"
        
        y_test, len_test = multi_pred_TFM(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx, x_stats, normalisation)
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
        
        
        x_tR = descale(x_test[0, -1, :4, 0], x_stats['mean'], x_stats['std'])
        y_tR = descale(y_test[0, 0, :4, 0], x_stats['mean'], x_stats['std'])
        
        x_tR = descale(x_test[0, -1, :, 0], x_stats['mean'], x_stats['std'])
        y_tR = descale(y_test[0, 0, :, 0], x_stats['mean'], x_stats['std'])
        print("Absolute error scaled:        ", MAE(x_test[0, step_idx + n_his, :, 0], y_test[0, 0, :, 0]))
        print("Absolute error de-scaled:   ", MAE(x_tR, y_tR))
        

        

        for ix in tmp_idx:
            te = evl[ix - 2:ix + 1]
            print(f'Time Step {ix + 1}: MAPE {te[0]:7.3%}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')
