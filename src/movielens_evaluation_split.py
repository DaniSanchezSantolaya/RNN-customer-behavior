import tensorflow as tf
import numpy as np
import os
# from PhasedLSTMCell_v1 import *
# from PhasedLSTMCell import *
import time
import sys
import pickle
from rnn_dynamic import *
# from rnn_attentional import * #For the attentional experiment
import scipy
from scipy import spatial
import pandas as pd

checkpoint_path = 'checkpoints/rep0-lstm2-256-1-128-adam-10000000000-20170516-161401/best_model/model_best.ckpt-3072000'

max_interactions = 100

# tensorflow model
model_parameters = {}
model_parameters['opt'] = 'adam'
model_parameters['learning_rate'] = 0.01
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
model_parameters['embedding_activation'] = 'linear'
model_parameters['y_length'] = 1
model_parameters['W_emb_init'] = 'None'
type_input = 'one-hot'
input_embeddings_size = 0
# Parameters for the attentional model only
model_parameters['attentional_layer'] = 'embedding'
model_parameters['attention_weights_activation'] = 'tanh'
model_parameters['init_stdev'] = 0.1

if type_input == 'one-hot':
    with open("pickles/movielens/X_test_" + str(max_interactions) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        X_test = pickle.load(handle)
    with open("pickles/movielens/Y_test_" + str(max_interactions) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        Y_test = pickle.load(handle)
elif type_input == 'embeddings':
    with open("pickles/movielens/X_test_" + str(max_interactions) + "_embeddings_" + str(
            input_embeddings_size) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        X_test = pickle.load(handle)
    with open("pickles/movielens/Y_test_" + str(max_interactions) + "_embeddings_" + str(
            input_embeddings_size) + "_2009_filter20_rep2.pickle", 'rb') as handle:
        Y_test = pickle.load(handle)

# model_parameters['n_input'] = X_test[0].toarray().shape[1]
# model_parameters['n_output'] = Y_test[0].toarray().shape[1]
# model_parameters['seq_length'] = X_test[0].toarray().shape[0]

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


# Create different splits according to the number of ocurrences
start_date_train = '2009-01-01'
date_test = '2014-10-01'
movies_min_ratings = 20
min_seq_length = 5
max_seq_length = 100
print('Create training samples')
# Ratings .dat format: UserID::MovieID::Rating::Timestamp
user_ids = []
movie_ids = []
ratings = []
timestamps = []
dataset_file = "../data/Movielens/ml-20m/ratings.csv"
df = pd.read_csv(dataset_file)
df['userId'] = df['userId'].astype(np.int32)
df['movieId'] = df['movieId'].astype(np.int32)
df['rating'] = df['rating'].astype(np.float16)
df['date'] = pd.to_datetime(df['timestamp'], unit='s')
df_date = df[df.date > start_date_train]
# Filter out movies with less than X ratings
filter_movies = df_date.movieId.value_counts()[(df_date.movieId.value_counts() > movies_min_ratings)].index.values
print(len(filter_movies))
df_date = df_date[df_date.movieId.isin(filter_movies)]
print('Total number of ratings:' + str(len(df_date)))
print('Number of different users: ' + str(len(df_date['userId'].unique())))
print('Number of different movies: ' + str(len(df_date['movieId'].unique())))
sys.stdout.flush()
# Filter out users with less than X ratings
filter_users = df_date['userId'].value_counts()[(df_date['userId'].value_counts() >= min_seq_length)].index.values
df_date = df_date[df_date.userId.isin(filter_users)]
print('Total number of ratings:' + str(len(df_date)))
print('Number of different users: ' + str(len(df_date['userId'].unique())))
print('Number of different movies: ' + str(len(df_date['movieId'].unique())))
sys.stdout.flush()

# Splits (by percentage, or by count value)
# max_count = df_date.movieId.value_counts().values[0]
# value_first_cut = max_count * 0.05
# value_second_cut = max_count * 0.1
value_first_cut = 100
value_second_cut = 1000
movies_first_cut = df_date.movieId.value_counts()[df_date.movieId.value_counts() < value_first_cut]
movies_second_cut = df_date.movieId.value_counts()[(df_date.movieId.value_counts() >= value_first_cut) &
                                                   (df_date.movieId.value_counts() < value_second_cut)]
movies_third_cut = df_date.movieId.value_counts()[df_date.movieId.value_counts() >= value_second_cut]
print('First cut value: ' + str(value_first_cut))
print('Second cut value: ' + str(value_second_cut))
print('First cut num movies: ' + str(len(movies_first_cut)))
print('Second cut num movies: ' + str(len(movies_second_cut)))
print('Third cut num movies: ' + str(len(movies_third_cut)))
print('Total movies: ' + str(len(movies_first_cut) + len(movies_second_cut) + len(movies_third_cut)))



def get_distances_output_embeddings(embedding_pred):
    distances = []
    for l in range(len(W_emb)):
        distances.append(scipy.spatial.distance.cosine(embedding_pred, W_emb[l, :]))
    return distances


def evaluate_sample(predictions, y_true, k, b_distances=False):
    # If predictions are probabilites, sort in reverse order, if are distances, sort from lower to higher
    reverse_sort = True
    if b_distances:
        reverse_sort = False
    idx_predictions = np.arange(len(predictions))
    sorted_pred, sorted_idx = zip(*sorted(zip(predictions, idx_predictions), reverse=reverse_sort))
    # Recall - Overall
    _, y_true_idx = np.where(y_true == 1)
    correct_idx = set(sorted_idx[:k]).intersection(set(y_true_idx))
    num_pos_k = len(correct_idx)
    total_pos = len(y_true)
    recall_user_k = (num_pos_k / float(total_pos))
    # Recall - per split
    total_pos_s1 = len(set(y_true_idx).intersection(set(movies_first_cut)))
    total_pos_s2 = len(set(y_true_idx).intersection((set(movies_second_cut))))
    total_pos_s3 = len(set(y_true_idx).intersection((set(movies_third_cut))))
    predicted_pos_s1 = len(set(correct_idx).intersection((set(movies_first_cut))))
    predicted_pos_s2 = len(set(correct_idx).intersection((set(movies_second_cut))))
    predicted_pos_s3 = len(set(correct_idx).intersection((set(movies_third_cut))))
    recall_user_k_s1=recall_user_k_s2=recall_user_k_s3= -1
    if total_pos_s1 > 0:
        recall_user_k_s1 = predicted_pos_s1 / float(total_pos_s1)
    if total_pos_s2 > 0:
        recall_user_k_s2 = predicted_pos_s2 / float(total_pos_s2)
    if total_pos_s3 > 0:
        recall_user_k_s3 = predicted_pos_s3 / float(total_pos_s3)

    # Precision - Overall
    precision_user_k = (num_pos_k) / float(k)
    # Precision per split
    precision_user_k_s1=precision_user_k_s2=precision_user_k_s3= -1
    if total_pos_s1 > 0:
        precision_user_k_s1 = predicted_pos_s1 / float(k)
    if total_pos_s2 > 0:
        precision_user_k_s2 = predicted_pos_s2 / float(k)
    if total_pos_s3 > 0:
        precision_user_k_s3 = predicted_pos_s3 / float(k)

    # Precision at R - Overall
    correct_idx_at_R = set(sorted_idx[:total_pos]).intersection(set(y_true_idx))
    precision_r = len(correct_idx_at_R) / float(total_pos)
    # Precision at R - Split - Skip for now

    # Sps - Overall
    first_movie = y_true_idx[0]
    if first_movie in sorted_idx[0:k]:
        sps_k = 1
    else:
        sps_k = 0
    # Sps - split
    sps_k_s1=sps_k_s2=sps_k_s3 = -1
    if first_movie in movies_first_cut:
        if first_movie in sorted_idx[0:k]:
            sps_k_s1 = 1
        else:
            sps_k_s1 = 0
    elif first_movie in movies_second_cut:
        if first_movie in sorted_idx[0:k]:
            sps_k_s2 = 1
        else:
            sps_k_s2 = 0
    elif first_movie in movies_first_cut:
        if first_movie in sorted_idx[0:k]:
            sps_k_s3 = 1
        else:
            sps_k_s3 = 0
    # Map
    sum_precisions = 0
    actual_pos = 0
    for i in range(k):
        if sorted_idx[i] in y_true_idx:
            actual_pos += 1  # CHECK THIS!! IT MAY BE NOT CORRECT
            sum_precisions += actual_pos / float(i + 1)
    ap_k = sum_precisions / min(k, len(y_true))

    return recall_user_k, precision_user_k, precision_r, sps_k, ap_k, num_pos_k, total_pos, total_pos_s1, total_pos_s2, total_pos_s3, predicted_pos_s1, predicted_pos_s2, predicted_pos_s3, recall_user_k_s1, recall_user_k_s2, recall_user_k_s3, precision_user_k_s1, precision_user_k_s2, precision_user_k_s3, sps_k_s1, sps_k_s2, sps_k_s3


# Make predictions in chunks

k = 10
recalls = []
precisions = []
precisions_r = []
spss = []
aps = []
num_poss = []
total_poss = []
# per splits
total_poss_s1 = []
total_poss_s2 = []
total_poss_s3 = []
predicted_poss_s1 = []
predicted_poss_s2 = []
predicted_poss_s3 = []
recalls_s1 = []
recalls_s2 = []
recalls_s3 = []
precisions_s1 = []
precisions_s2 = []
precisions_s3 = []
spss_s1 = []
spss_s2 = []
spss_s3 = []


num_movies = Y_test[0].toarray().shape[1]
batch_size = 100
for i in range(0, len(X_test), batch_size):
    if type_input == 'one-hot':
        x_test = [x.toarray() for x in X_test[i:i + batch_size]]
        y_test = [y.toarray() for y in Y_test[i:i + batch_size]]
    elif type_input == 'embeddings':
        x_test = [x for x in X_test[i:i + batch_size]]
        y_test = [y.toarray() for y in Y_test[i:i + batch_size]]
    logits, y_pred = model.predict(x_test, checkpoint_path)
    for j in range(len(y_pred)):
        if model_parameters['type_output'] == 'embeddings':
            predictions = get_distances_output_embeddings(y_pred[j, :])
            b_distances = True
        else:
            predictions = y_pred[j]
            b_distances = False
            recall_user_k, precision_user_k, precision_r, sps_k, ap_k, num_pos_k, total_pos, total_pos_s1, total_pos_s2, total_pos_s3, predicted_pos_s1, predicted_pos_s2, predicted_pos_s3, recall_user_k_s1, recall_user_k_s2, recall_user_k_s3, precision_user_k_s1, precision_user_k_s2, precision_user_k_s3, sps_k_s1, sps_k_s2, sps_k_s3 = evaluate_sample(predictions,
                                                                                                          y_test[j], k,
                                                                                                          b_distances)
        recalls.append(recall_user_k)
        precisions.append(precision_user_k)
        precisions_r.append(precision_r)
        spss.append(sps_k)
        aps.append(ap_k)
        num_poss.append(num_pos_k)
        total_poss.append(total_pos)
        # per splits
        total_poss_s1.append(total_pos_s1)
        total_poss_s2.append(total_pos_s2)
        total_poss_s3.append(total_pos_s3)
        predicted_poss_s1.append(predicted_pos_s1)
        predicted_poss_s2.append(predicted_pos_s2)
        predicted_poss_s3.append(predicted_pos_s3)
        if recall_user_k_s1 != -1:
            recalls_s1.append(recall_user_k_s1)
        if recall_user_k_s2 != -1:
            recalls_s2.append(recall_user_k_s2)
        if recall_user_k_s3 != -1:
            recalls_s3.append(recall_user_k_s3)
        if precision_user_k_s1 != -1:
            precisions_s1.append(precision_user_k_s1)
        if precision_user_k_s2 != -1:
            precisions_s2.append(precision_user_k_s2)
        if precision_user_k_s3 != -1:
            precisions_s3.append(precision_user_k_s3)
        if sps_k_s1 != -1:
            spss_s1.append(sps_k_s1)
        if sps_k_s2 != -1:
            spss_s2.append(sps_k_s2)
        if sps_k_s3 != -1:
            spss_s3.append(sps_k_s3)


    print(str(i) + '/' + str(len(X_test)))

print('Mean recall users: ' + str(np.mean(recalls)))
print('Mean precisions users: ' + str(np.mean(precisions)))
print('Mean precisions@r users: ' + str(np.mean(precisions_r)))
print('Mean spss: ' + str(np.mean(spss)))
print('MAP: ' + str(np.mean(aps)))
total_recall = np.sum(num_poss) / float(np.sum(total_poss))
print('Total Recall (no mean recall users): ' + str(total_recall))
# per splits
print('---')
print('PERFORMANCE PER SPLITS')
total_recall_s1 = np.sum(predicted_poss_s1) / float(np.sum(total_poss_s1))
total_recall_s2 = np.sum(predicted_poss_s2) / float(np.sum(total_poss_s2))
total_recall_s3 = np.sum(predicted_poss_s3) / float(np.sum(total_poss_s3))
print('Total positives s1: ' + str(np.sum(total_poss_s1)))
print('Predicted positives s1: ' + str(np.sum(predicted_poss_s1)))
print('Total Recall s1: ' + str(total_recall_s1))
print('Total positives s2: ' + str(np.sum(total_poss_s2)))
print('Predicted positives s2: ' + str(np.sum(predicted_poss_s2)))
print('Total Recall s2: ' + str(total_recall_s2))
print('Total positives s3: ' + str(np.sum(total_poss_s3)))
print('Predicted positives s3: ' + str(np.sum(predicted_poss_s3)))
print('Total Recall s3: ' + str(total_recall_s3))
print('Mean recall users s1: ' + str(np.mean(recalls_s1)))
print('Mean recall users s2: ' + str(np.mean(recalls_s2)))
print('Mean recall users s3: ' + str(np.mean(recalls_s3)))
print('Mean precisions users s1: ' + str(np.mean(precisions_s1)))
print('Mean precisions users s2: ' + str(np.mean(precisions_s2)))
print('Mean precisions users s3: ' + str(np.mean(precisions_s3)))
print('Mean sps users s1: ' + str(np.mean(spss_s1)))
print('Mean sps users s2: ' + str(np.mean(spss_s2)))
print('Mean sps users s3: ' + str(np.mean(spss_s3)))

