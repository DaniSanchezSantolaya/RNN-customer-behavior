import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pickle
import os
from preprocess_santander import * 
from dataset import *
#from rnn_static import *
from rnn_dynamic import *
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
import ast
from sklearn.manifold import TSNE
import pylab as plt
from collections import Counter
import seaborn
import gc

random.seed(17)
random.seed(17)
np.random.seed(17)


#python experiment.py representation max_interactions padding b_load_pickles p_val opt learning_rate n_hidden batch_size rnn_type rnn_layers dropout l2_reg type_output max_steps load_df_pickle ks
#python experiment.py 4 10 right False 0.2 adam 0.0001 64 128 lstm 1 0.1 0.0 sigmoid 40000 True [2,3,4,5,6,7]
#python experiment.py 4 5 left True 0.1 adam 0.0001 128 128 lstm2 1 0.0 0.0 sigmoid 2500000 True [2,3,4,5,6,7]
#python experiment.py 4 5 left False 0.1 adam 0.0001 128 128 lstm2 1 0.0 0.0 sigmoid 2500000 True [2,3,4,5,6,7]
#python experiment.py 4 6 right True 0.1 adam 0.0001 128 128 lstm 1 0.1 0.0 sigmoid 1000000 True [2,3,4,5,6,7]




target_columns = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
           'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
           'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
           'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
           'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
           'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

           
           
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

aux_features = [ 'ind_empleado_A', 'ind_empleado_B', 'ind_empleado_F', 'ind_empleado_N', 'ind_empleado_S',
					'sexo_H', 'sexo_V',
					'ind_nuevo_1', 'ind_nuevo_0',
					'antiguedadnormalized',
					'indrel_1.0', 'indrel_99.0',
					'indrel_1mes_1', 'indrel_1mes_2', 'indrel_1mes_3', 'indrel_1mes_4', 'indrel_1mes_5',
					'indresi_N', 'indresi_S', 
					'indext_N', 'indext_S',
					'indfall_N', 'indfall_S',
					'ind_actividad_cliente_0', 'ind_actividad_cliente_1',
					'rentanormalized',
					'segmento_01 - TOP', 'segmento_02 - part', 'segmento_03 - UNIVERSITARIO'
					]
					
				
aux_features = []

if len(sys.argv) < 2: #default #C:\Projects\Thesis\src>python experiment.py 4 6 right True 0.1 adam 0.0001 16 128 lstm2 1 0.1 0.01 sigmoid 30000 True
    representation = 4
    max_interactions = 6
    padding = 'right'
    b_load_pickles = True
    p_val = 0.1


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

    load_df_pickle = True
    k = 7

else:
 
    #Representation parameters
    representation = int(sys.argv[1])
    max_interactions = int(sys.argv[2])
    padding = sys.argv[3]
    b_load_pickles = str2bool(sys.argv[4])
    p_val = float(sys.argv[5])


    #Model and training parameters
    model_parameters = {}
    model_parameters['opt'] = sys.argv[6]
    model_parameters['learning_rate'] = float(sys.argv[7])
    model_parameters['n_hidden'] = int(sys.argv[8])
    model_parameters['batch_size'] = int(sys.argv[9])
    model_parameters['rnn_type'] = sys.argv[10]
    model_parameters['rnn_layers'] = int(sys.argv[11])
    model_parameters['dropout'] = float(sys.argv[12])
    model_parameters['l2_reg'] = float(sys.argv[13])
    model_parameters['type_output'] = sys.argv[14]
    model_parameters['max_steps'] = int(sys.argv[15])
    model_parameters['padding'] = padding

    load_df_pickle = sys.argv[16]
    ks =  ast.literal_eval(sys.argv[17])



name_submission = 'kaggle_submissions/rep_' +str(representation) + '-interactions_' + str(max_interactions) + '-padding_' + str(padding) + '-' + model_parameters['opt'] + '-lrate_' + str(model_parameters['learning_rate']) + '-hidden_' + str(model_parameters['n_hidden']) + '-bSize_' + str(model_parameters['batch_size']) + '-' + model_parameters['rnn_type'] + '-rnn_layers' + str(model_parameters['rnn_layers']) + '-dropout_' + str(model_parameters['dropout']) + '-L2_' + str(model_parameters['l2_reg']) + '-typeoutput_' + str(model_parameters['type_output']) + '-max_steps_' + str(model_parameters['max_steps']) 
if len(aux_features) > 0:
    name_submission = name_submission + '_aux_features.csv'
else:
    name_submission = name_submission + '.csv'

print('Arguments: ')
print('representation: ' + str(representation))
print('max_interactions: ' + str(max_interactions))
print('padding: ' + str(padding))
print('Load pickle: ' + str(b_load_pickles))
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
print('load_df_pickle: ' + str(load_df_pickle))


#### Load pickle
def load_pickles():
    aux_features_length = str(len(aux_features))
    with open('pickles/X_train_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length +  '.pickle', 'rb') as handle:
        X_train = pickle.load(handle)
    with open('pickles/X_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    with open('pickles/Y_train_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'rb') as handle:
        Y_train = pickle.load(handle)
    with open('pickles/X_local_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'rb') as handle:
        X_local_test = pickle.load(handle)
    with open('pickles/Y_local_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'rb') as handle:
        Y_local_test = pickle.load(handle)

    return X_train, Y_train, X_test, X_local_test, Y_local_test
    
def save_pickles(X_train, Y_train, X_test, X_local_test, Y_local_test):
    #Save pickle
    aux_features_length = str(len(aux_features))
    with open('pickles/X_train_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/X_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/Y_train_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'wb') as handle:
        pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/X_local_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'wb') as handle:
        pickle.dump(X_local_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/Y_local_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '.pickle', 'wb') as handle:
        pickle.dump(Y_local_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


        
def generate_validation_set(X_train, Y_train, X_test):
    print('Initial X_train size: ' + str( len(X_train)))
    print('Initial Y_train size: ' + str( len(Y_train)))
    print('Initial X_test size: ' + str( len(X_test)))

    num_val = int(len(X_train) * p_val)

    indices = np.random.permutation(len(X_train))
    val_idx, training_idx = indices[:num_val], indices[num_val:]
    X_val, X_train = X_train[val_idx], X_train[training_idx]
    Y_val, Y_train  = Y_train[val_idx], Y_train[training_idx]

    
    #Transform validatoin format to np.array
    X_val2 = []
    Y_val2 = []
    for x,y in zip(X_val, Y_val):
        X_val2.append(x.toarray())
        Y_val2.append(y.toarray().reshape(y.toarray().shape[1]))
    X_val = np.array(X_val2)
    Y_val = np.array(Y_val2)


    print('Final X_train size: ' + str( len(X_train)))
    print('Final Y_train size: ' + str( len(Y_train)))
    print('Final X_val size: ' + str( len(X_val)))
    print('Final Y_val size: ' + str( len(Y_val)))
    print('Final X_test size: ' + str( len(X_test)))

    return X_train, X_val, X_test, Y_train, Y_val        
        
df_test = load_test_csv()
if b_load_pickles:
    print('Load pickles')
    X_train, Y_train, X_test, X_local_test, Y_local_test = load_pickles()
else:
    print('Build pickles')
    if load_df_pickle:
        df = load_train_from_pickle_interactions()
    else:
        df = load_train_csv()
    X_train, Y_train, X_test, X_local_test, Y_local_test = build_train_and_test(df, df_test, representation, max_interactions, aux_features, padding)
    save_pickles(X_train, Y_train, X_test, X_local_test, Y_local_test)



X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train, X_val, X_test, Y_train, Y_val = generate_validation_set(X_train, Y_train, X_test)

model_parameters['n_input'] = X_train[0].toarray().shape[1]
model_parameters['n_output'] = Y_train[0].toarray().shape[1]
model_parameters['seq_length'] = X_train[0].toarray().shape[0]
print('num features: ' + str(model_parameters['n_input']))
print('seq length: ' + str(model_parameters['seq_length']))
print('num output: ' + str(model_parameters['n_output']))
ds = DataSet(X_train, Y_train, X_val, Y_val)
X_train = []
Y_train = []
X_val = []
Y_val = []


#Create tensorflow model
model = RNN_dynamic(model_parameters)
model.create_model()
model.train(ds)



X_local_test = np.array(X_local_test)
Y_local_test = np.array(Y_local_test)


#TSN
ds._X_train = []
ds._Y_train = []
ds._X_test = []
import gc
gc.collect()

X_tsn = []
labels_tsn = []


if representation == 2:
    pass
elif representation==4:
    for x in X_local_test:
        num_interactions = np.count_nonzero(np.sum(x, axis=1) > 0)
        if num_interactions == 1:
            X_tsn.append(x)

        
    #hidden_states = np.zeros((len(X_local_test), model_parameters['n_output']))

    X_tsn = np.array(X_tsn)
    hidden_states = model.get_last_hidden_state(X_tsn)
    for i in range(len(X_tsn)):
        label = X_tsn[i][0].tolist().index(1)
        labels_tsn.append(label)
    print('Total points with 1 interaction:')
    print(Counter(labels_tsn))
    #
    model_tsn = TSNE(n_components=2, random_state=0)
    Y = model_tsn.fit_transform(hidden_states)
    
    df_plot = pd.DataFrame(dict(dim1=Y[:, 0], dim2=Y[:,1], color=labels_tsn))
    sns.lmplot('dim1', 'dim2', data=df_plot, hue='color', fit_reg=False)
    plt.show()
    
    plt.scatter(Y[:, 0], Y[:,1], len(set(labels_tsn)), labels_tsn)
    plt.title('t-SNE representation when only one interaction ')
    plt.show()
    
    model_tsn = []
    Y = []
    gc.collect()
 
''' 
    #Plot 2: for recibo
    X_tsn = X_local_test
    hidden_states = model.get_last_hidden_state(X_tsn)
    model_tsn_2 = TSNE(n_components=2, random_state=0)
    indices = np.random.permutation(len(hidden_states))
    idx = indices[:5000]
    hidden_states = hidden_states[idx,:]
    Y2 = model_tsn_2.fit_transform(hidden_states)
    labels_tsn_2 = []
    for i in range(len(Y2)):
        recibo_adds = np.sum(X_tsn[i, :, 23])
        recibo_drops = np.sum(X_tsn[i, :, 23 + 24])
        if (recibo_adds + recibo_drops) > 0:
            labels_tsn_2.append(0)
        else:
            labels_tsn_2.append(1)
    print('Points with recibo and points with no recibo:')
    print(Counter(labels_tsn_2))
    df_plot = pd.DataFrame(dict(dim1=Y2[:, 0], dim2=Y2[:,1], color=labels_tsn_2))
    sns.lmplot('dim1', 'dim2', data=df_plot, hue='color', fit_reg=False)
    #seaborn.regplot(Y2[:, 0], Y2[:,1], label=labels_tsn_2)
    plt.show()

    #plt.scatter(Y2[:, 0], Y2[:,1], colors=labels_tsn_2)
    #plt.title('t-SNE representation products with interaction in recibo and with no interaction in recibo ')
    #plt.show()
'''   
   
'''    
    #Plot 3
    X_tsn = X_local_test
    hidden_states = model.get_last_hidden_state(X_tsn)
    model_tsn_2 = TSNE(n_components=2, random_state=0)
    indices = np.random.permutation(len(hidden_states))
    idx = indices[:5000]
    hidden_states = hidden_states[idx,:]
    Y2 = model_tsn_2.fit_transform(hidden_states)
    labels_tsn_3 = []
    for i in range(len(Y2)):
        recibo_adds = np.sum(X_tsn[i, :, 2])
        recibo_drops = np.sum(X_tsn[i, :, 26])
        if (recibo_adds + recibo_drops) > 0:
            labels_tsn_3.append(0)
        else:
            labels_tsn_3.append(1)

    print('Points with recibo and points with no recibo:')
    print(Counter(labels_tsn_3))
    df_plot = pd.DataFrame(dict(dim1=Y2[:, 0], dim2=Y2[:,1], color=labels_tsn_3))
    sns.lmplot('dim1', 'dim2', data=df_plot, hue='color', fit_reg=False)
    #seaborn.regplot(Y2[:, 0], Y2[:,1], label=labels_tsn_2)
    plt.show()
'''    
    
'''
    #Plot 4, separate by number of interactions
    X_tsn = X_local_test
    labels_tsn_4 = []
    for x in X_local_test[:5000]:
        num_interactions = np.count_nonzero(np.sum(x, axis=1) > 0)
        if num_interactions == 1:
            labels_tsn_4.append(1)
        elif num_interactions == 2:
            labels_tsn_4.append(2)
        elif num_interactions == 3:
            labels_tsn_4.append(3)
        elif num_interactions == 4:
            labels_tsn_4.append(4)
        elif num_interactions == 5:
            labels_tsn_4.append(5)
        if num_interactions == 0:
            labels_tsn_4.append(0)
            
    print(Counter(labels_tsn_4))
    df_plot = pd.DataFrame(dict(dim1=Y2[:, 0], dim2=Y2[:,1], color=labels_tsn_4))
    sns.lmplot('dim1', 'dim2', data=df_plot, hue='color', fit_reg=False)
    plt.show()
 '''   
    
    #Plot 5, interactions with only one interaction in the last interaction
    X_tsn = []
    labels_tsn_5 = []
    for x in X_local_test:
        num_interactions = np.count_nonzero(np.sum(x[-1]) > 0)
        if num_interactions == 1:
            X_tsn.append(x)
            
    X_tsn = np.array(X_tsn)
    hidden_states = model.get_last_hidden_state(X_tsn)
    for i in range(len(X_tsn)):
        label = X_tsn[i,-1].tolist().index(1)
        labels_tsn_5.append(label)
    print('Total points with 1 interaction:')
    print(Counter(labels_tsn_5))
    #
    model_tsn = TSNE(n_components=2, random_state=0)
    Y = model_tsn.fit_transform(hidden_states)
    
    df_plot = pd.DataFrame(dict(dim1=Y[:, 0], dim2=Y[:,1], color=labels_tsn_5))
    sns.lmplot('dim1', 'dim2', data=df_plot, hue='color', fit_reg=False)
    plt.show()
              
    print(Counter(labels_tsn_5))
    df_plot = pd.DataFrame(dict(dim1=Y2[:, 0], dim2=Y2[:,1], color=labels_tsn_5))
    sns.lmplot('dim1', 'dim2', data=df_plot, hue='color', fit_reg=False)
    plt.show()
    
    
    print('end')
    
    
    
    



    



'''
plt.scatter(Y2[:, 0], Y2[:,1], 2, labels_tsn_2)
plt.title('t-SNE representation products with interaction in recibo and with no interaction in recibo ')
plt.show()
'''
   
       





