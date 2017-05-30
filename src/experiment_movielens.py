import numpy as np
import pandas as pd
import random
import pickle
import os
#from rnn_static import *
from rnn_dynamic import *
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
import ast
import time

print(time.strftime("%Y%m%d-%H%M%S"))

random.seed(17)
random.seed(17)
np.random.seed(17)

#python experiment_movielens.py max_interactions padding p_val opt learning_rate n_hidden batch_size rnn_type rnn_layers dropout l2_reg type_output max_steps init_stdev embedding_size embedding_activation type_input input_embedding_size W_emb_init representation
#python experiment_movielens.py 100 right 0.03 adam 0.01 50 128 lstm2 1 0.2 0 softmax 4000000 0.1 0 linear one-hot > movielens.txt
#ubuntu@packer-ubuntu-16:~$ python3.5 experiment_movielens.py 100 right 0.025 adam 0.01 50 128 lstm2 1 0.2 0 softmax 10000000 0.1 0 linear one-hot > movielens.txt

# Change if using dataset dynamic
num_total_files = 1#71
num_validation_file = 8
year = '2009' #2009

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
    model_parameters['init_stdev'] = 1
    model_parameters['embedding_size'] = 0
    model_parameters['embedding_activation'] = 'linear'
    model_parameters['W_emb_init'] = None


    # Only used for reading the corresponding pickle
    type_input = 'one-hot'
    input_embedding_size = 0

    # representation: 1: 1 sample per user, 2: data augmentation, 3: intermediate errors
    representation = 3
    


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
    model_parameters['init_stdev'] = float(sys.argv[14])
    model_parameters['embedding_size'] = int(sys.argv[15])
    model_parameters['embedding_activation'] = sys.argv[16]
    model_parameters['W_emb_init'] = sys.argv[19]


    # Only used for reading the corresponding pickle
    type_input = sys.argv[17]
    input_embedding_size = sys.argv[18]

    # representation: 1: 1 sample per user, 2: data augmentation, 3: intermediate errors
    representation = int(sys.argv[20])

if representation == 3:
    model_parameters['y_length'] = max_interactions
else:
    model_parameters['y_length'] = 1


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
print('init_stdev: ' + str(model_parameters['init_stdev']))
print('embedding_size: ' + str(model_parameters['embedding_size']))
print('embedding_activation: ' + str(model_parameters['embedding_activation']))
print('type_input: ' + str(type_input))
print('input_embedding_size: ' + str(input_embedding_size))
print('W_emb_init: ' + str(model_parameters['W_emb_init']))

last_part_filename = ""
b_output_embeddings = False
if model_parameters['type_output'] == 'embeddings':
    last_part_filename = '_output'
    b_output_embeddings = True

if num_total_files == 1:
    #### Load train pickle
    if type_input == 'one-hot':
        with open("pickles/movielens/X_train_" + str(max_interactions) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'rb') as handle:
            X_train = pickle.load(handle)
        with open("pickles/movielens/Y_train_" + str(max_interactions) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'rb') as handle:
            Y_train = pickle.load(handle)
    elif type_input == 'embeddings':
        with open("pickles/movielens/X_train_" + str(max_interactions) + "_embeddings_" + str(input_embedding_size) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'rb') as handle:
            X_train = pickle.load(handle)
        with open("pickles/movielens/Y_train_" + str(max_interactions) + "_embeddings_" + str(input_embedding_size) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'rb') as handle:
            Y_train = pickle.load(handle)
else: #Load first file, just for get the input, output sizes used later
    with open("pickles/movielens/X_train_" + str(max_interactions) + "_embeddings_" + str(
            input_embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file0" + last_part_filename + ".pickle", 'rb') as handle:
        X_train = pickle.load(handle)

    with open("pickles/movielens/Y_train_" + str(max_interactions) + "_embeddings_" + str(
            input_embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file0" + last_part_filename + ".pickle", 'rb') as handle:
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
     
if type_input == 'one-hot':
    model_parameters['n_input'] = X_train[0].toarray().shape[1]
    model_parameters['n_output'] = Y_train[0].toarray().shape[1]
    model_parameters['seq_length'] = X_train[0].toarray().shape[0]
else:
    model_parameters['n_input'] = X_train[0].shape[1]
    if model_parameters['type_output'] == 'embeddings':
        model_parameters['n_output'] = Y_train[0].shape[0]
    else:
        print(Y_train[0].shape)
        model_parameters['n_output'] = Y_train[0].shape[1]
    model_parameters['seq_length'] = X_train[0].shape[0]
print('num features: ' + str(model_parameters['n_input']))
print('seq length: ' + str(model_parameters['seq_length']))
print('num output: ' + str(model_parameters['n_output']))
if num_total_files == 1:
    from dataset import *
    ds = DataSet(X_train, Y_train, X_val, Y_val, [], [], representation, [], [], name_dataset = 'movielens')
else:
    from dataset_dynamic import *
    ds = DataSet(max_interactions, input_embedding_size, year, representation, num_total_files, num_validation_file, b_output_embeddings, name_dataset='movielens')
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
