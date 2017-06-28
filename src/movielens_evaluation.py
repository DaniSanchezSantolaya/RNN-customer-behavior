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
import scipy
from scipy import spatial



checkpoint_path = 'checkpoints/rep2-lstm2-256-1-128-adam-10000000000-20170623-223914/best_model/model_best.ckpt-13158400'
research_question_code = 'test'
remove_already_rated = True

max_interactions = 100


# tensorflow model
model_parameters = {}
model_parameters['opt'] = 'adam'
model_parameters['learning_rate'] = 0.001
model_parameters['n_hidden'] = 256
model_parameters['batch_size'] = 128
model_parameters['rnn_type'] = 'lstm2'
model_parameters['rnn_layers'] = 1
model_parameters['dropout'] = 0.0
model_parameters['l2_reg'] = 0.0
model_parameters['type_output'] = 'softmax'
model_parameters['max_steps'] = 3000000
model_parameters['padding'] = 'right'
model_parameters['embedding_size'] = 64
model_parameters['embedding_activation'] = 'tanh'
model_parameters['y_length'] = 1
model_parameters['W_emb_init'] = 'None'
type_input = 'one-hot'
input_embeddings_size = 0
# Parameters for the attentional model only
model_parameters['attentional_layer'] = 'hidden_state'
model_parameters['attention_weights_activation'] = 'tanh'
model_parameters['init_stdev'] = 0.1

if type_input == 'one-hot':
    with open("pickles/movielens/X_test_" + str(max_interactions) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        X_test = pickle.load(handle)
    with open("pickles/movielens/Y_test_" + str(max_interactions) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        Y_test = pickle.load(handle)
elif type_input == 'embeddings':
    with open("pickles/movielens/X_test_" + str(max_interactions) + "_embeddings_" + str(input_embeddings_size) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        X_test = pickle.load(handle)
    with open("pickles/movielens/Y_test_" + str(max_interactions) + "_embeddings_" + str(input_embeddings_size) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        Y_test = pickle.load(handle)
    with open("pickles/movielens/X_test_" + str(max_interactions) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        X_test2 = pickle.load(handle)

#model_parameters['n_input'] = X_test[0].toarray().shape[1]
#model_parameters['n_output'] = Y_test[0].toarray().shape[1]
#model_parameters['seq_length'] = X_test[0].toarray().shape[0]

if type_input == 'one-hot':
    model_parameters['n_input'] = X_test[0].toarray().shape[1]
    model_parameters['n_output'] = Y_test[0].toarray().shape[1]
    model_parameters['seq_length'] = X_test[0].toarray().shape[0]
else:
    model_parameters['n_input'] = X_test[0].shape[1]
    if model_parameters['type_output'] == 'embeddings':
        model_parameters['n_output'] = input_embeddings_size
        with open("pickles/movielens/" + "W_emb_64" + ".pickle", 'rb') as handle:
            W_emb = pickle.load(handle)
            W_emb = W_emb.astype(np.float32)
    else:
        model_parameters['n_output'] = Y_test[0].shape[1]
    model_parameters['seq_length'] = X_test[0].shape[0]

print('Input dimension: ' + str(model_parameters['n_input']))
print('Output dimension: ' + str(model_parameters['n_output']))

# Create tensorflow model and train
print('Create model...')
model = RNN_dynamic(model_parameters)
model.create_model()

def get_distances_output_embeddings(embedding_pred):
    distances = []
    for l in range(len(W_emb)):
        distances.append(scipy.spatial.distance.cosine(embedding_pred, W_emb[l, :]))
    return distances

def evaluate_sample(predictions, l_already_watched, y_true, k, b_distances=False):
    # If predictions are probabilites, sort in reverse order, if are distances, sort from lower to higher
    reverse_sort = True
    if b_distances:
        reverse_sort = False
    idx_predictions = np.arange(len(predictions))
    sorted_pred, sorted_idx = zip(*sorted(zip(predictions, idx_predictions), reverse=reverse_sort))
    # Remove movies already rated
    if remove_already_rated:
        sorted_idx = [x for x in sorted_idx if x not in l_already_watched]
    # Recall
    _, y_true_idx = np.where(y_true ==1)
    correct_idx = set(sorted_idx[:k]).intersection(set(y_true_idx))
    num_pos_k = len(correct_idx)
    total_pos = len(y_true)
    recall_user_k = (num_pos_k/float(total_pos))
    # Precision
    precision_user_k = (num_pos_k)/float(k)
    # Precision at R
    correct_idx_at_R = set(sorted_idx[:total_pos]).intersection(set(y_true_idx))
    precision_r = len(correct_idx_at_R) / float(total_pos)
    # Sps
    first_movie = y_true_idx[0]
    if first_movie in sorted_idx[0:k]:
        sps_k = 1
    else:
        sps_k = 0
    # Map
    sum_precisions = 0
    actual_pos = 0
    for i in range(k):
        if sorted_idx[i] in y_true_idx:
            actual_pos += 1 #CHECK THIS!! IT MAY BE NOT CORRECT
            sum_precisions += actual_pos/float(i+1)
    ap_k = sum_precisions/min(k, len(y_true))
    
    return recall_user_k, precision_user_k, precision_r, sps_k, ap_k, num_pos_k, total_pos

    
# Make predictions in chunks

k = 10
recalls = []
precisions = []
precisions_r = []
spss = []
aps = []
num_poss = []
total_poss = []

num_movies = Y_test[0].toarray().shape[1]
batch_size = 100
for i in range(0, len(X_test), batch_size):
    if type_input == 'one-hot':
        x_test = [x.toarray() for x in X_test[i:i+batch_size]]
        y_test = [y.toarray() for y in Y_test[i:i+batch_size]]
    elif type_input == 'embeddings':
        x_test = [x for x in X_test[i:i + batch_size]]
        y_test = [y.toarray() for y in Y_test[i:i + batch_size]]
        x_test2 = [x.toarray() for x in X_test2[i:i+batch_size]]
    logits, y_pred = model.predict(x_test, checkpoint_path)
    for j in range(len(y_pred)):
        if model_parameters['type_output'] == 'embeddings':
            predictions = get_distances_output_embeddings(y_pred[j, :])
            b_distances = True
        else:
            predictions = y_pred[j]
            b_distances = False
        if type_input == 'one-hot':
            _, l_already_watched = np.where(x_test[j] == 1)
        elif type_input == 'embeddings':
            _, l_already_watched = np.where(x_test2[j] == 1)
        recall_user_k, precision_user_k, precision_r, sps_k, ap_k, num_pos_k, total_pos = evaluate_sample(predictions, l_already_watched,
                                                                                                          y_test[j], k, b_distances)
        recalls.append(recall_user_k)
        precisions.append(precision_user_k)
        precisions_r.append(precision_r)
        spss.append(sps_k)
        aps.append(ap_k)
        num_poss.append(num_pos_k)
        total_poss.append(total_pos)
    print(str(i) + '/' + str(len(X_test)))

print('Mean recall users: ' + str(np.mean(recalls)))
print('Mean precisions users: ' + str(np.mean(precisions)))
print('Mean precisions@r users: ' + str(np.mean(precisions_r)))
print('Mean spss: ' + str(np.mean(spss)))
print('MAP: ' + str(np.mean(aps)))
total_recall = np.sum(num_poss)/float(np.sum(total_poss))
print('Total Recall (no mean recall users): ' + str(total_recall))

# Save pickles of precisions_r and spss
with open("pickles/movielens/measures/precision_r_" + research_question_code + ".pickle", 'wb') as handle:
    pickle.dump(precisions_r, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("pickles/movielens/measures/spss_" + research_question_code + ".pickle", 'wb') as handle:
    pickle.dump(spss, handle, protocol=pickle.HIGHEST_PROTOCOL)

