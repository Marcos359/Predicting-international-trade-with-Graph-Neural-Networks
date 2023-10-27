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

# Arreglo para que tensorflow funcione en v1, ya que parece que todo se ha hecho en esa versión.
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
base = 256 # 128
## REGLAS = [1*, B, C*] [C*, D, E]
#  B y D <=~ 512
#  Tendencias: MAE = A y RMSE  = A*(4-5)
#  La versión ascente parece ser la mejor!
blocks = [[1, 512, 1024], [1024, 1024, 2048]]
# [[1, 512, 1024], [1024, 512, 2048]] #  [[1, 1260, 2048]] # [[1, 512, 1024], [1024, 1024, 2048]]
#[[1, base, base*2], [base*2, base, base*4]] # ASCENDIENTE muy eficaz con base = 256
# [[1, 32, 64], [64, 32, 128]] # ESTANDAR

# Mejor combinación:
   # n_his = 13, n_pred = 3 y blocks = [[1, 512, 1024], [1024, 512, 2028]]
   # Mejor --lr = 1e-3
   # Mejor --ks = 4
   # Normalisation = robust
   # opt = ADAM
   # Parece ser que lo único que importa es que el final sea lo más grande posible!

## COSAS QUE PROBAR
# ESTRUCTURA DESCENDIENTE
# MÁS BATCH SIZE
# n_pred = 1 y  n_his = 15
# opt = ADAM
# Normalisation = robust
# 
# ¿Realizamos una transformación logaritmica a las variables?

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'Pesos_TODO_4.csv')) # o  el Pesos 3  autorregresiva exigente, 
# Pesos_TODO_4.csv, parece  ser peor que  Pesos_TODO_REDUCIDO.csv
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))
    
# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st apprsox - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)

##################################
# Get the total size of the tensor's content in bytes
# total_size_bytes = Lk.shape[0] * Lk.shape[1] * 8 # Lk.dtype.size

#if total_size_bytes > 2 * 1024 * 1024 * 1024:
#    print("The tensor's content is larger than 2GB.")
#else:
#    print("The tensor's content is smaller or equal to 2GB.")
##################################

tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
# n_his + n_pred == n_frame: Es la cantidad de filas que vamos a meter en cada serie, ahora mismo 18. 
# n_train + n_val + n_test == longitud_"dia" - n_frame; para este caso un "dia" es toda la base, luego su longitud es 26.
# Datos_TODO_REDUCIDO
#data_file = 'Datos_TODO_reducido_interpolado_2.csv'
data_file = 'Datos_TODO_reducido_interpolado_2.csv'
n_train, n_val, n_test = 20, 0, 1 # 8, 2, 1 # 18, 0, 3 # 11, 9, 1 # 18, 0, 3  
# 20, 0, 1 Se prueva 2020!
# 16, 0, 1 Se prueva 2016! 
# 6, 0, 1 Se prueva 2016!
interpolation = True
PeMS = data_gen_simple(pjoin('./dataset', data_file), (n_train, n_val, n_test, args.normalisation), n, n_his + n_pred, interpolation)
try:
    print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
except:
    print("Dataset column by column, or an error has appeared, be sure to check!")
    

### CON ESTO ENTRENARIA CON 3 SET TRAINS DE 5 AÑOS CADA UNO
### Y LUEGO SE HARIA LA VALIDACIÓN SE HARIA CON UN SET DE 1 Y EL TEST CON UN SOLO SET DE 1

############### CONTROL DEL GRAFO ###############
print(PeMS.get_data("test").shape)
#df = pd.DataFrame(PeMS.get_data("train"))
#df.to_csv('file1.csv')
############### ERROR DE CONTROL ###############
#raise ValueError('Se ha llegado hasta el final!')


if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode, args.normalisation)