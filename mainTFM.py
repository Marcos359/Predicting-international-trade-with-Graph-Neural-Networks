# @Time     : Jan. 02, 2019 22:17
# @Author   : Veritas YIN
# @FileName : main.py
# @Version  : 1.0
# @Project  : Orion
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

import os
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin


import tensorflow
from tensorflow.compat import v1 as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utilsTFM import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=2450) #default=228
parser.add_argument('--n_his', type=int, default=13) #default=13
parser.add_argument('--n_pred', type=int, default=3) #default=3
parser.add_argument('--batch_size', type=int, default=1) # default=50 antes 1, con el batch size de 2
parser.add_argument('--epoch', type=int, default= 100) # default=50
parser.add_argument('--save', type=int, default=25)
parser.add_argument('--ks', type=int, default=4) # 4
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3) #1e-3)
parser.add_argument('--opt', type=str, default='ADAM') # RMSProp
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge') ## 'sep' para casos de n_pred < 3, pero eso luego da problemas!
parser.add_argument('--normalisation', type=str, default='robust') # z_score o robust o log_scale o none

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design

blocks = [[1, 512, 1024], [1024, 1024, 2048]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'Pesos_TODO_4.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))
    
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st apprsox - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)

tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
# n_his + n_pred == n_frame: It is the number of rows we are going to put in each series, right now 18. 
# n_train + n_val + n_test == longitud_"dia" - n_frame; For this case a "day" is the whole base, so its length is 26.

data_file = 'Datos_TODO_reducido_interpolado_2.csv'
n_train, n_val, n_test = 20, 0, 1
# If you are using interpolation:
# 20, 0, 1 You are testing 2020!
# 16, 0, 1 You are testing 2016!

# If you are not using interpolation:
# 6, 0, 1 You are testing 2016!

interpolation = True
PeMS = data_gen_simple(pjoin('./dataset', data_file), (n_train, n_val, n_test, args.normalisation), n, n_his + n_pred, interpolation)
try:
    print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
except:
    print("Dataset column by column, or an error has appeared, be sure to check!")


print(PeMS.get_data("test").shape)



if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, args.normalisation)
