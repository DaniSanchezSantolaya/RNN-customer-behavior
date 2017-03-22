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
model = RNN_static(model_parameters)
model.create_model()
model.train(ds)
#for rep4 obtain only test samples with interactions
if representation == 4:
    indices_interactions = []
    for i in range(len(X_test)):
        if np.count_nonzero(X_test[i]) > 0:
            indices_interactions.append(i)
    #val_idx, training_idx = indices[:num_val], indices[num_val:]
    #X_val, X_train = X_train[val_idx], X_train[training_idx]
    #Y_val, Y_train  = Y_train[val_idx], Y_train[training_idx]
    pred_test = np.zeros((len(X_test), model_parameters['n_output']))
    pred_test[indices_interactions] = model.predict(X_test[indices_interactions])
else:
    pred_test = model.predict(X_test)


def compute_most_added_products():
    #Compute the number of times that a product is added in all portfolio
    adds_per_product = np.zeros(len(target_columns))
    grouped = df.groupby('ncodpers')
    i = 0
    for name,group in grouped:
        for j in range(len(target_columns)):
            product = target_columns[j]
            value_counts = group[product].diff().value_counts()
            if 1.0 in value_counts:
                adds_per_product[j] += value_counts[1.0]
        i += 1
        if i % 50000 == 0:
            print(i)
    sorted_freq, sorted_prods_total = zip(*sorted(zip(adds_per_product, target_columns), reverse=True ))  
    for adds, prod in zip(sorted_freq, sorted_prods_total):
        print(prod + ': ' + str(adds))
        

    with open('pickles/sorted_freq.pickle', 'wb') as handle:
        pickle.dump(sorted_freq, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/sorted_prods_total.pickle', 'wb') as handle:
        pickle.dump(sorted_prods_total, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return sorted_freq, sorted_prods_total


def compute_products_before_test_time():
    if not 'df' in locals():
        print('Load df')
        df = load_train_csv()
    grouped = df.groupby('ncodpers')
    dict_last_values = {}
    a = 0
    for ncodpers,group in grouped:
        dict_last_values[ncodpers] = group[target_columns].values[-1]
        if a % 50000 == 0:
            print(a)
        a += 1
    with open('pickles/dict_last_values.pickle', 'wb') as handle:
        pickle.dump(dict_last_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dict_last_values
    
    
    

    
#Generate predictions file
if (os.path.isfile('pickles/sorted_freq.pickle')) and (os.path.isfile('pickles/sorted_prods_total.pickle')):
    print('Load sorted prods')
    with open('pickles/sorted_freq.pickle', 'rb') as handle:
        sorted_freq = pickle.load(handle)
    with open('pickles/sorted_prods_total.pickle', 'rb') as handle:
        sorted_prods_total = pickle.load(handle)
else:
	sorted_freq, sorted_prods_total = compute_most_added_products()
	
print(os.path.isfile('pickles/dict_last_values.pickle'))
if (os.path.isfile('pickles/dict_last_values.pickle')):
    with open('pickles/dict_last_values.pickle', 'rb') as handle:
        dict_last_values = pickle.load(handle)
else:
    dict_last_values = compute_products_before_test_time()
		
#Prediction only added but until seven items + Baseline most added products when no interactions 
ncodpers_test = df_test['ncodpers']


f = open(name_submission, 'w')
f.write('ncodpers,added_products\n')

if (representation == 1) or (representation == 2):
    for i in range(len(ncodpers_test)):
        sorted_pred, sorted_prods = zip(*sorted(zip(pred_test[i], target_columns), reverse=True ))  
        f.write(str(ncodpers_test[i]) + ',')  # python will convert \n to os.linesep
        num_added = 0
        for prob,prod in zip(sorted_pred, sorted_prods):
            #Check if the product was already added
            idx_prod = target_columns.index(prod)
            if X_test[i][-1][idx_prod] == 0:
                f.write(prod + ' ')
                num_added += 1
                if num_added == 7:
                    break
        f.write('\n')
        if i % 100000 == 0:
            print(i)
else:

    pred_int = 0
    pred_no_int = 0
    for i in range(len(ncodpers_test)):

        f.write(str(ncodpers_test[i]) + ',')  # python will convert \n to os.linesep
        
        #last_values = df[df.ncodpers == ncodpers_test[i]][target_columns].values[-1] #Slow, probably better build a dictionary
        last_values = dict_last_values[ncodpers_test[i]]
        
        #If contains interactions we use the model
        if (1 in X_test[i]) or (-1 in X_test[i]): #Check
            pred_int = pred_int + 1
            num_added = 0
            sorted_pred, sorted_prods = zip(*sorted(zip(pred_test[i], target_columns), reverse=True ))  
            for prob,prod in zip(sorted_pred, sorted_prods):
                #Check if the product was already added - FIX, now it's different
                idx_prod = target_columns.index(prod)
                if last_values[idx_prod] == 0:
                    f.write(prod + ' ')
                    num_added += 1
                    if num_added == 7:
                        break
                        
        else: #if there is no interactions, we use the baseline most added products
            pred_no_int = pred_no_int + 1
            num_added = 0
            for prod in sorted_prods_total:
                #Check if the product was already added - FIX, now it's different
                idx_prod = target_columns.index(prod)
                if last_values[idx_prod] == 0:
                    f.write(prod + ' ')
                    num_added += 1
                    if num_added == 7:
                        break
        f.write('\n')
        if i % 100000 == 0:
            print(i)
    print('Predictions with interactions: ' + str(pred_int))
    print('Predictions with no interactions: ' + str(pred_no_int))

f.close()  




#Evaluate on local test set   
total_no_interactions = 0
array_ordered = np.arange(1, 25)
def evaluate_sample(predictions, y_true, k):
    global total_no_interactions
    global array_ordered
    sorted_pred, sorted_y = zip(*sorted(zip(predictions, y_true[0]), reverse=True )) #TODO: Discard the products that were already part of the portfolio in the last step
    #Recall at k
    true_pos_k = sum(sorted_y[0:k])
    num_pos = sum(y_true[0])
    recall_user = (true_pos_k/float(num_pos))
    #Map at k
    precisions = sorted_y/array_ordered
    sum_precisions = np.sum(precisions[:k])
    if num_pos == 0:
        map_k = 0
        recall_user = 0
        total_no_interactions += 1
    else:
        map_k = sum_precisions/float(min(num_pos, k))
        #map_k = sum_precisions/float(k)



    return recall_user, true_pos_k, num_pos, map_k

X_local_test = np.array(X_local_test)
Y_local_test = np.array(Y_local_test)

recalls_model = []
recalls_baseline = []
maps_model = []
maps_baseline = []
for k in ks:
    recall_users = []
    map_k_users = []
    total_true_pos_k = 0
    total_pos = 0
    if representation == 2:
        pred_local_test = model.predict(X_local_test)
        
        for i in range(len(pred_local_test)):
            #sorted_pred, sorted_y = zip(*sorted(zip(pred_local_test[i], X_local_test[i]), reverse=True ))
            recall_user, true_pos_k, num_pos, map_k = evaluate_sample(pred_local_test[i], Y_local_test[i], k)
            map_k_users.append(map_k)
            recall_users.append(recall_user)
            total_true_pos_k += true_pos_k
            total_pos += num_pos
    elif representation==4:
        print('Local test rep 4')
        pred_local_test = np.zeros((len(X_local_test), model_parameters['n_output']))
        pred_local_test = model.predict(X_local_test)
        
       
        for i in range(len(X_local_test)):  
            #sorted_pred, sorted_y = zip(*sorted(zip(pred_local_test[i], Y_local_test[i]), reverse=True )) #TODO: Discard the products that were already part of the portfolio in the last step
            recall_user, true_pos_k, num_pos, map_k = evaluate_sample(pred_local_test[i], Y_local_test[i], k)
            map_k_users.append(map_k)
            recall_users.append(recall_user)
            total_true_pos_k += true_pos_k
            total_pos += num_pos
            if i % 100000 == 0:
                print(i)

    print('-------------------')            
    print('Results local test for k = ' + str(k) + ':')
    print('Total users evaluated: ' + str(len(recall_users)))
    print('Total True positives at k: :' + str(total_true_pos_k))
    print('Total True positives: ' + str(total_pos))
    recall_k = total_true_pos_k/ float(total_pos)
    print('Total Recall at ' + str(k) + ': ' + str(recall_k))
    print('Mean recall at ' + str(k) + ' by user: ' + str(np.mean(recall_users)))
    print('Mean map at ' + str(k) + ' by user: ' + str(np.mean(map_k_users)))
    #print('Max map at ' + str(k) + ' by user: ' + str(np.max(map_k_users)))   
    #print('Min map at ' + str(k) + ' by user: ' + str(np.min(map_k_users))) 
    print('Total no interactions: ' + str(total_no_interactions))
    recalls_model.append(np.mean(recall_users))
    maps_model.append(np.mean(map_k_users))

    

    #Most added products baseline
    target_freq = np.zeros(24)
    for i in range(len(sorted_freq)):
        idx_product = target_columns.index(sorted_prods_total[i])
        prob_product = sorted_freq[i]
        target_freq[idx_product] = prob_product
        
    recall_users = []
    map_k_users = []
    total_true_pos_k = 0
    total_pos = 0
    for i in range(len(pred_local_test)):
        recall_user, true_pos_k, num_pos, map_k = evaluate_sample(target_freq, Y_local_test[i], k)
        map_k_users.append(map_k)
        recall_users.append(recall_user)
        total_true_pos_k += true_pos_k
        total_pos += num_pos

    print('')
    print('Results local test BASELINE:')
    print('Total users evaluated: ' + str(len(recall_users)))
    print('Total True positives at k: :' + str(total_true_pos_k))
    print('Total True positives: ' + str(total_pos))
    recall_k = total_true_pos_k/ float(total_pos)
    print('Total Recall at ' + str(k) + ': ' + str(recall_k))
    print('Mean recall at ' + str(k) + ' by user: ' + str(np.mean(recall_users)))
    print('Mean map at ' + str(k) + ' by user: ' + str(np.mean(map_k_users)))
    #print('Max map at ' + str(k) + ' by user: ' + str(np.max(map_k_users)))   
    #print('Min map at ' + str(k) + ' by user: ' + str(np.min(map_k_users)))
    recalls_baseline.append(np.mean(recall_users))
    maps_baseline.append(np.mean(map_k_users))


#ML baseline
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators=25)
#clf.fit(X_train_valid, y_train_valid)
#clf_probs = clf.predict_proba(X_test)
    
#make plot 
import matplotlib.pyplot as plt

plt.plot(recalls_model, linestyle='-', marker='o', label='recall_k_model')
plt.plot(recalls_baseline, linestyle='-', marker='o', label='recall_k_freq_baseline')
plt.legend()
plt.ylim([0,1])
plt.show()

plt.plot(maps_model, linestyle='-', marker='o', label='map_k_model')
plt.plot(maps_baseline, linestyle='-', marker='o', label='map_k_freq_baseline')
plt.legend()
plt.ylim([0,1])
plt.show()


