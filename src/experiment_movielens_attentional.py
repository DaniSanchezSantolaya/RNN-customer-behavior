import numpy as np
import pandas as pd
import random
import pickle
import os
from dataset import *
#from rnn_static import *
from rnn_attentional import *
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
import ast
import time


random.seed(17)
random.seed(17)
np.random.seed(17)

#python experiment_movielens_attentional.py max_interactions padding p_val opt learning_rate n_hidden batch_size rnn_type rnn_layers dropout l2_reg type_output max_steps max_steps embedding_size attentional_layer embedding_activation attention_weights_activation type_input input_embedding_size



start = time.time()

           
           
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


if len(sys.argv) < 2:

    max_interactions = 100
    padding = 'right'
    b_load_pickles = True
    p_val = 0.15


    #Model and training parameters
    model_parameters = {}
    model_parameters['opt'] = 'adam'
    model_parameters['learning_rate'] = 0.001
    model_parameters['n_hidden'] = 16
    model_parameters['batch_size'] = 128
    model_parameters['rnn_type'] = 'lstm2'
    model_parameters['rnn_layers'] = 1
    model_parameters['dropout'] = 0.1
    model_parameters['l2_reg'] = 0.01
    model_parameters['type_output'] = 'sigmoid'
    model_parameters['max_steps'] = 30000
    model_parameters['padding'] = padding
    model_parameters['embedding_size'] = 16
    model_parameters['attentional_layer'] = 'embedding'
    model_parameters['embedding_activation'] = 'linear'
    model_parameters['attention_weights_activation'] = 'linear'
    model_parameters['type_input'] = 'one-hot'
    model_parameters['input_embedding_size'] = 0


else:

    #Representation parameters
    max_interactions = int(sys.argv[1])
    padding = sys.argv[2]
    p_val = float(sys.argv[3])


    #Model and training parameters
    model_parameters = {}
    model_parameters['opt'] = sys.argv[4]
    model_parameters['learning_rate'] = float(sys.argv[5])
    model_parameters['n_hidden'] = int(sys.argv[6])
    model_parameters['batch_size'] = int(sys.argv[7])
    model_parameters['rnn_type'] = sys.argv[8]
    model_parameters['rnn_layers'] = int(sys.argv[9])
    model_parameters['dropout'] = float(sys.argv[10])
    model_parameters['l2_reg'] = float(sys.argv[11])
    model_parameters['type_output'] = sys.argv[12]
    model_parameters['max_steps'] = int(sys.argv[13])
    model_parameters['padding'] = padding
    model_parameters['embedding_size'] = int(sys.argv[14])
    model_parameters['attentional_layer'] = str(sys.argv[15])
    model_parameters['embedding_activation'] = str(sys.argv[16])
    model_parameters['attention_weights_activation'] = str(sys.argv[17])
    model_parameters['type_input'] = sys.argv[18]
    model_parameters['input_embedding_size'] = sys.argv[19]





print('Arguments: ')
print('max_interactions: ' + str(max_interactions))
print('padding: ' + str(padding))
print('p_val: ' + str(p_val))
print('opt: ' + str(model_parameters['opt']))
print('learning_rate: ' + str(model_parameters['learning_rate']))
print('n_hidden: ' + str(model_parameters['n_hidden']))
print('batch_size: ' + str(model_parameters['batch_size']))
print('rnn_type: ' + str(model_parameters['rnn_type']))
print('rnn_layers: ' + str(model_parameters['rnn_layers']))
print('dropout: ' + str(model_parameters['dropout']))
print('l2_reg: ' + str(model_parameters['l2_reg']))
print('type_output: ' + str(model_parameters['type_output']))
print('max_steps: ' + str(model_parameters['max_steps']))
print('embedding_size: ' + str(model_parameters['embedding_size']))
print('attentional_layer: ' + str(model_parameters['attentional_layer']))
print('type_input: ' + str(model_parameters['type_input']))


#### Load train pickle

#### Load train pickle
if model_parameters['type_input'] == 'one-hot':
    with open("pickles/movielens/X_train_" + str(max_interactions) + "_2009_filter20.pickle", 'rb') as handle:
        X_train = pickle.load(handle)
    with open("pickles/movielens/Y_train_" + str(max_interactions) + "_2009_filter20.pickle", 'rb') as handle:
        Y_train = pickle.load(handle)
elif model_parameters['type_input'] == 'embeddings':
    with open("pickles/movielens/X_train_" + str(max_interactions) + "_embeddings_" + str(model_parameters['input_embedding_size']) + "_2009_filter20.pickle", 'rb') as handle:
        X_train = pickle.load(handle)
    with open("pickles/movielens/Y_train_" + str(max_interactions) + "_embeddings_" + str(model_parameters['input_embedding_size']) + "_2009_filter20.pickle", 'rb') as handle:
        Y_train = pickle.load(handle)


X_train = np.array(X_train)
Y_train = np.array(Y_train)
    
# Split train in  train/validation
num_val = int(len(X_train) * p_val)

indices = np.random.permutation(len(X_train))
val_idx, training_idx = indices[:num_val], indices[num_val:]
X_val, X_train = X_train[val_idx], X_train[training_idx]
Y_val, Y_train  = Y_train[val_idx], Y_train[training_idx]
#Transform validatoin format to np.array
#X_val2 = []
#Y_val2 = []
#for x,y in zip(X_val, Y_val):
    #X_val2.append(x.toarray())
    #Y_val2.append(y.toarray().reshape(y.toarray().shape[1]))
#X_val = np.array(X_val2)
#Y_val = np.array(Y_val2)


print('Final X_train size: ' + str( len(X_train)))
print('Final Y_train size: ' + str( len(Y_train)))
print('Final X_val size: ' + str( len(X_val)))
print('Final Y_val size: ' + str( len(Y_val)))
     

model_parameters['n_input'] = X_train[0].toarray().shape[1]
model_parameters['n_output'] = Y_train[0].toarray().shape[1]
model_parameters['seq_length'] = X_train[0].toarray().shape[0]
print('num features: ' + str(model_parameters['n_input']))
print('seq length: ' + str(model_parameters['seq_length']))
print('num output: ' + str(model_parameters['n_output']))
ds = DataSet(X_train, Y_train, X_val, Y_val, [], [], 0, [], [], name_dataset = 'movielens')
X_train = []
Y_train = []
X_val = []
Y_val = []


#Create tensorflow model
model = RNN_dynamic(model_parameters)
model.create_model()
model.train(ds)

    
end = time.time()

print('Script time: ' + str(start - end))
