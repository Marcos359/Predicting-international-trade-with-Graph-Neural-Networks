# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

# Arreglo para que tensorflow funcione en v1, ya que parece que todo se ha hecho en esa versión. + 
import tensorflow
from tensorflow.compat import v1 as tf

import numpy as np
import time

## Añadido!
import pandas as pd


def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt

    # Arreglo para quitar el modo "eager" + 
    tf.disable_eager_execution()

    # Placeholder for model training, x is  empthy!
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Define model loss
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.get_collection('copy_loss'))
    tf.summary.scalar('copy_loss', copy_loss)  

    ############### ERROR DE CONTROL ###############
    #print("############### CONTROL de Pred")
    #print(train_loss)
    #print(pred)
    #raise ValueError('Se ha llegado hasta el final!')
            
            
    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e20, 1e20, 1e20]) #np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge': # Por defecto
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e20, 1e20, 1e20] * len(step_idx)) #np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        # This is where the training happens
        for i in range(epoch):
            start_time = time.time()
            # x_batch son simplemente trozos de train de tamaño ["batch_size", 21, 228, 1]
            # Además, el bucle se va a ejecutar len(inputs.get_data('train'))/batch_size, que para el caso estandar es 182-183 veces.
            # n_his,hace referencia a la canrtidad de registros historicos que se van a utilizar para la predicción, por defecto se utilizan 13 de los posibles 21 registros posibles.
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                #Descubrir que es merged y train_op.
                # Train_op = Train optimizers
                # merged = Más parámetros de configuración o seguimiento?
             
                
                # Entra datos para hacer predicciónes.
                # Calcula el train_loss con las X (originalmente 8) filas que no utiliza.
                # Como se mete el "train_op", aquí es donde se realiza la retropropagación, porque para eso necesitas un optimizador!!
                # Recordatorio, de que el último de los x es utilizado como el y, es decir que si de [0:n_his + 1] van trece, 
                    # se utilizarán 12 para realizar una predicción y esta se comparará al treceabo restante para calcular el loss.
                summary, w = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1})
                writer.add_summary(summary, i * epoch_step + j)
                
                
                ############## CONTROL ###############
                #print("############### CONTROL ###############")
                #print(x_batch.shape)
                #print(x_batch[0, 0:n_his + 1, 0, 0])
                #print("############### CONTROL ###############")
                #print(w)
                #print(j)
                #print(n_his + 1)
                ############### ERROR DE CONTROL ###############
                #raise ValueError('Se ha llegado hasta el final!')
                
                
                # Cada 50 predicciones, se ve el progreso.
                if j % 50 == 0: # if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')

            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')
                
            start_time = time.time()
            min_va_val, min_val = \
                model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)
            
            #print("############### CONTROL ###############")
            #print([min_va_val, min_val])
            #print("############### CONTROL ###############")
            
            # tmp_idx = n_pred - 1 
            # De momento como no supera la validación minima de 40, no sale. 
            for ix in tmp_idx:
                va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                print(f'Time Step {ix + 1}: '
                      f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')
            
            ############### CONTROL ###############
            # create a scalar summary
            summary = tf.Summary(value=[tf.Summary.Value(tag='Validation Loss', simple_value=min_va_val[2])])

            # add the summary to the writer
            writer.add_summary(summary, global_step=i)
            ############### CONTROL ###############

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, 'STGCN')
        writer.flush()
        writer.close()
    print('Training model finished!')
