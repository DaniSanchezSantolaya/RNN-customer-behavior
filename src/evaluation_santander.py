import tensorflow as tf
import numpy as np
import os
#from PhasedLSTMCell_v1 import *
#from PhasedLSTMCell import *
import time
import sys
import pickle
from rnn_dynamic import *
#from rnn_attentional import * #For the attentional experiment


representation = 9
max_interactions = 20
padding = 'right'
aux_features_length = '0'
time_column = 'time_from_last_interaction'

with open('pickles/X_local_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '_' + time_column + '.pickle', 'rb') as handle:
    X_test = pickle.load(handle)

with open('pickles/Y_local_test_rep' + str(representation) + '_' + str(max_interactions) + '_' + padding + '_' + aux_features_length + '_' + time_column + '.pickle', 'rb') as handle:
    Y_test = pickle.load(handle)


checkpoint_path = 'checkpoints/rep9-lstm2-128-1-128-adam-200000000-20170515-122036/last_model/last_model.ckpt-6649600'



# tensorflow model
model_parameters = {}
model_parameters['opt'] = 'adam'
model_parameters['learning_rate'] = 0.001
model_parameters['n_hidden'] = 128
model_parameters['batch_size'] = 128
model_parameters['rnn_type'] = 'lstm2'
model_parameters['rnn_layers'] = 1
model_parameters['dropout'] = 0.0
model_parameters['l2_reg'] = 0.0
model_parameters['type_output'] = 'sigmoid'
model_parameters['max_steps'] = 3000000
model_parameters['padding'] = 'right'
model_parameters['n_input'] = X_test[0].toarray().shape[1]
model_parameters['n_output'] = Y_test[0].toarray().shape[1]
model_parameters['seq_length'] = X_test[0].toarray().shape[0]
model_parameters['embedding_size'] = 0
# Parameters for the attentional model only
model_parameters['attentional_layer'] = 'hidden_state'
model_parameters['init_stdev'] = 0.1
model_parameters['embedding_activation'] = 'linear'
model_parameters['attention_weights_activation'] = 'linear'
if representation == 4:
    model_parameters['y_length'] = 1
elif representation == 9:
    model_parameters['y_length'] = max_interactions

# Create tensorflow model and train
print('Create model...')
model = RNN_dynamic(model_parameters)
model.create_model()

total_no_interactions = 0
array_ordered = np.arange(1, 25)


def evaluate_sample(predictions, y_true, k):
    global total_no_interactions
    global array_ordered
    sorted_pred, sorted_y = zip(*sorted(zip(predictions, y_true), reverse=True))
    #Recall at k
    true_pos_k = sum(sorted_y[0:k])
    num_pos = sum(y_true)
    recall_user = (true_pos_k/float(num_pos))
    #Map at k
    precisions = np.cumsum(sorted_y)/array_ordered
    precisions = precisions * sorted_y
    sum_precisions = np.sum(precisions[:k])
    if num_pos == 0:
        map_k = 0
        recall_user = 0
        total_no_interactions += 1
    else:
        map_k = sum_precisions/float(min(num_pos, k))
        #map_k = sum_precisions/float(k)




    return recall_user, true_pos_k, num_pos, map_k

# Make predictions in chunks

k = 3
recalls = []
spss = []
aps = []
num_poss = []
total_poss = []

num_movies = Y_test[0].toarray().shape[1]
batch_size = 100
for i in range(0, len(X_test), batch_size):
    x_test = [x.toarray() for x in X_test[i:i + batch_size]]
    y_test = [y.toarray() for y in Y_test[i:i + batch_size]]
    logits, y_pred = model.predict(x_test, checkpoint_path)
    for j in range(len(y_pred)):
        #recall_user_k, ap_k, num_pos_k, total_pos = evaluate_sample(y_pred[j], y_test[j], k)
        recall_user, true_pos_k, num_pos, map_k = evaluate_sample(y_pred[j], y_test[j][0], k)
        recalls.append(recall_user)
        aps.append(map_k)
        num_poss.append(true_pos_k)
        total_poss.append(num_pos)
    print(str(i) + '/' + str(len(X_test)))


print('Mean recall users: ' + str(np.mean(recalls)))
print('MAP: ' + str(np.mean(aps)))
total_recall = np.sum(num_poss) / float(np.sum(total_poss))
print('Total Recall (no mean recall users): ' + str(total_recall))