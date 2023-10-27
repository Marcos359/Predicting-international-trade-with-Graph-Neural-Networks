# @Time     : Jan. 12, 2019 19:01
# @Author   : Veritas YIN
# @FileName : base_model.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from models.layers import *
from os.path import join as pjoin

# Arreglo para que tensorflow funcione en v1, ya que parece que todo se ha hecho en esa versión. + 
import tensorflow
from tensorflow.compat import v1 as tf


def build_model(inputs, n_his, Ks, Kt, blocks, keep_prob):
    '''
    Build the base model.
    :param inputs: placeholder.
    :param n_his: int, size of historical records for training.
    :param Ks: int, kernel size of spatial convolution.
    :param Kt: int, kernel size of temporal convolution.
    :param blocks: list, channel configs of st_conv blocks.
    :param keep_prob: placeholder.
    '''
    x = inputs[:, 0:n_his, :, :]
    
    ######## CONTROL ########
    #print(1, "Valor de x: ", x)
    ######## CONTROL ########
    
    # Ko>0: kernel size of temporal convolution in the output layer.
    # Aquí se establece en kernel size!
    Ko = n_his
    # ST-Block
    for i, channels in enumerate(blocks):
        x = st_conv_block(x, Ks, Kt, channels, i, keep_prob, act_func='GLU')
        print("Nuevo x tras un bloque convolucional: ", x)
        Ko -= 2 * (Kt - 1)
        
    ######## CONTROL ########
    #print(2)
    ######## CONTROL ########

    # Output Layer
    if Ko > 1:
        y = output_layer(x, Ko, 'output_layer')
    else:
        raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{Ko}".')

    ######## CONTROL ########
    #print(3)
    ######## CONTROL ########
    
    # Calcula el error MAPE
    error_mape = tf.reduce_mean(tf.abs((inputs[:, n_his:n_his + 1, :, :] - inputs[:, n_his - 1:n_his, :, :]) / inputs[:, n_his - 1:n_his, :, :] + 1000))
    
    #diff = tf.abs(inputs[:, n_his:n_his + 1, :, :] - inputs[:, n_his - 1:n_his, :, :])
    #divisor = inputs[:, n_his - 1:n_his, :, :]
    #error = tf.where(tf.equal(divisor, 0), tf.zeros_like(diff), diff / divisor)
    #error_mape = tf.reduce_mean(error)
    
    # Esta expresión compara dos pasos de tiempo adyacentes en las entradas, es decir si hay una variación del loss en las iteraciones.
    tf.add_to_collection(name='copy_loss',
                         value=tf.nn.l2_loss(inputs[:, n_his - 1:n_his, :, :] - inputs[:, n_his:n_his + 1, :, :]))
    
    #No estrenar con MAPE!!!!
    
    # train_loss es la función de perdida, que es la del MSE.
    # Esta expresión compara las etiquetas de salida con un solo paso de tiempo específico de las entradas.
    # Supongo que con esto se hace la retropropagación, se compara lo que ha entrado con el siguiente paso del historial de forma implicita.
    # Siii, porque en el entrenamiento entra: inputs[:, 0:n_his+1, :, :], pero si luego solo se usa inputs[:, 0:n_his, :, :] como la x para predecir cosas perooo se utilizan todos luego el que queda para ver el error en training inputs[:, n_his:n_his + 1, :, :] y se compara con la prección anterior para ver como se ha modificado el loss en el tiempo usando.
    # La entrada solo tiene n_hist-1, para meter datos, y el último se queda como y, por eso no especifica la y directamente sino que utiliza el último de x como y, supongo que para usar un sistema muy custom, pero joder que chapucero y poco elegante parece.
    train_loss = tf.nn.l2_loss(y - inputs[:, n_his:n_his + 1, :, :])
    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    
    ######## CONTROL ########
    #print(4)
    ######## CONTROL ########
    
    return train_loss, single_pred


def model_save(sess, global_steps, model_name, save_path='./output/models/'):
    '''
    Save the checkpoint of trained model.
    :param sess: tf.Session().
    :param global_steps: tensor, record the global step of training in epochs.
    :param model_name: str, the name of saved model.
    :param save_path: str, the path of saved model.
    :return:
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, pjoin(save_path, model_name), global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')
