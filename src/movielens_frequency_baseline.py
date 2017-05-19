
import numpy as np
import os
import time
import sys
import pickle
import pandas as pd

# load test data
max_interactions = 100

with open("pickles/movielens/X_test_" + str(max_interactions) + "_2009_filter20_rep3.pickle", 'rb') as handle:
    X_test = pickle.load(handle)
with open("pickles/movielens/Y_test_" + str(max_interactions) + "_2009_filter20_rep3.pickle", 'rb') as handle:
    Y_test = pickle.load(handle)


# load movielens dataset

start_date_train = '2009-01-01'
date_test = '2014-10-01'
min_seq_length = 5
max_seq_length = 100

movies_min_ratings = 20  # 20

# representation: 1: 1 sample per user, 2: data augmentation, 3: intermediate errors
representation = 3
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
with open("../data/Movielens/ml-20m/df_date.pickle", 'wb') as handle:
    pickle.dump(df_date, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Filter out movies with less than X ratings
filter_movies = df_date.movieId.value_counts()[(df_date.movieId.value_counts() > movies_min_ratings)].index.values
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

# Build array mapping movie_id --> position in one-hot encoding
movieIds = np.zeros(max(df_date['movieId'].unique()) + 1, np.uint16)
i = 0
for movieId in df_date['movieId'].unique():
    movieIds[movieId] = i
    i += 1


# Build frequency predictions
prediction_array = np.zeros(len(df_date['movieId'].unique()))
counts_movies = df_date.movieId.value_counts()
counts_movies_id = counts_movies.index.values
counts_movies_c = counts_movies.values
total_ratings = float(counts_movies_c.sum())
for movie_id, movie_count in zip(counts_movies_id, counts_movies_c):
    pos_one_hot = movieIds[movie_id]
    prediction_array[pos_one_hot] = movie_count/total_ratings




def evaluate_sample(predictions, y_true, k):
    idx_predictions = np.arange(len(predictions))
    sorted_pred, sorted_idx = zip(*sorted(zip(predictions, idx_predictions), reverse=True))
    # Recall
    _, y_true_idx = np.where(y_true == 1)
    correct_idx = set(sorted_idx[:k]).intersection(set(y_true_idx))
    num_pos_k = len(correct_idx)
    total_pos = len(y_true)
    recall_user_k = (num_pos_k / float(total_pos))
    # Precision
    precision_user_k = (num_pos_k) / float(k)
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
            actual_pos += 1  # CHECK THIS!! IT MAY BE NOT CORRECT
            sum_precisions += actual_pos / float(i + 1)
    ap_k = sum_precisions / min(k, len(y_true))

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
    x_test = [x.toarray() for x in X_test[i:i + batch_size]]
    y_test = [y.toarray() for y in Y_test[i:i + batch_size]]
    for j in range(len(y_test)):
        recall_user_k, precision_user_k, precision_r, sps_k, ap_k, num_pos_k, total_pos = evaluate_sample(prediction_array, y_test[j], k)
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
total_recall = np.sum(num_poss) / float(np.sum(total_poss))
print('Total Recall (no mean recall users): ' + str(total_recall))