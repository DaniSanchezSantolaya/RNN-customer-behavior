import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import gc
import sys

load_pickle = False

start_date_train = '2009-01-01'
date_test = '2014-10-01'
min_seq_length = 5
max_seq_length = 100


movies_min_ratings = 20


if load_pickle:
    with open("../data/Movielens/ml-20m/df_date.pickle", 'rb') as handle:
        df_date = pickle.load(handle)
else:
    #Ratings .dat format: UserID::MovieID::Rating::Timestamp
    user_ids = []
    movie_ids = []
    ratings = []
    timestamps = []
    dataset_file = "../data/Movielens/ml-20m/ratings.csv"
    df = pd.read_csv(dataset_file)
    df['userId'] = df['userId'].astype(np.int32)
    df['movieId'] = df['movieId'].astype(np.int32)
    df['rating'] = df['rating'].astype(np.float16)
    df['date'] = pd.to_datetime(df['timestamp'],unit='s')

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

    


dtype_sparse = np.int8


seq_lengths = []
num_diff_items = len(df_date.movieId.unique())
grouped = df_date.groupby('userId')

X_train = []
Y_train = []
X_test = []
Y_test = []
user_train = []
user_test = []
discarded_users = []
i = 0
for name, group in grouped:
    
    # Obtain movies_ids and its position in the one-hot vector
    user_movie_ids_train = group[group['date'] < date_test].sort_values('date').movieId.values
    one_hot_pos_train = movieIds[user_movie_ids_train]
    num_movies_train = len(one_hot_pos_train)
       
    
    # Only add users with min_seq_length movies rated
    if num_movies_train > min_seq_length:
        ratings = np.zeros((num_movies_train, num_diff_items), dtype=np.int8)
        
        # Set 1 in the correspondent timestamp and one hot positions for every movie
        time_idx = np.array(range(0, num_movies_train))
        ratings[time_idx, one_hot_pos_train] = 1
            
        # Create training sample
        for j in range(min_seq_length, num_movies_train):
            x_train = np.zeros((max_seq_length, num_diff_items), dtype=np.int8)
            start = max(0, (j-max_seq_length))
            x_train[0:(j-start),:] = ratings[start:j,:]
            y_train = ratings[j, :]
            y_train = y_train

            X_train.append(sparse.csr_matrix(x_train, dtype=dtype_sparse))
            Y_train.append(sparse.csr_matrix(y_train, dtype=dtype_sparse))
            user_train.append(name)
            
        # Create sample for test
        x_test = np.zeros((max_seq_length, num_diff_items), dtype=np.int8)
        num_movies_sample = min(max_seq_length, num_movies_train)
        x_test[0:num_movies_sample,:] = ratings[-num_movies_sample:,:]
        
        user_movies_ids_test = group[group['date'] >= date_test].sort_values('date').movieId.values
        one_hot_pos_test = movieIds[user_movies_ids_test]
        num_movies_test = len(one_hot_pos_test)
        
        if num_movies_test > 0:
            user_test.append(name)
            time_idx = np.array(range(0, num_movies_test))
            y_test = np.zeros((num_movies_test, num_diff_items), dtype=np.int8)
            y_test[time_idx, one_hot_pos_test] = 1 
            
            X_test.append(sparse.csr_matrix(x_test, dtype=dtype_sparse))
            Y_test.append(sparse.csr_matrix(y_test, dtype=dtype_sparse))
    else:
        discarded_users.append(name)
    i += 1
    if i % 2500 == 0:
        print(i)
        print('Num training samples: ' + str(len(X_train)))
        sys.stdout.flush()

print('Num train samples: ' + str(len(X_train)))
print('Num test samples: ' + str(len(X_test)))
        
# Save pickles
with open("pickles/movielens/discarded_users_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(discarded_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("pickles/movielens/user_train_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(user_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("pickles/movielens/user_test_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(user_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
df_date = []
df = []
df_2015 = []
discarded_users = []
user_train = []
user_test = []
gc.collect()
with open("pickles/movielens/X_test_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("pickles/movielens/Y_test_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
X_test = []
Y_test = []
gc.collect()
with open("pickles/movielens/Y_train_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
Y_train = []
gc.collect()
with open("pickles/movielens/X_train_" + str(max_seq_length) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved pickles!')


    
