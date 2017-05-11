from gensim.models import word2vec
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import gc
import sys
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

load_pretrained_embeddings = False
embedding_size = 64
start_date_train = '2014-01-01'
date_test = '2014-10-01'
movies_min_ratings = 20
min_seq_length = 5
max_seq_length = 100
word2vec_iter = 5

# STEP 1: LOAD PRETRAINED EMBEDDINGS OR TRAIN THEM FROM SCRATCH

if load_pretrained_embeddings:
    word2vec = gensim.models.Word2Vec.load("word2vec_" + str(embedding_size) + ".txt")
else:

    start_date_embeddings = '2012-01-01'

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

    df_date = df[df.date > start_date_embeddings]

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

    grouped = df_date.groupby('userId')

    movie_sequences = []
    for name, group in grouped:
        movie_sequences.append(group.sort_values(['rating', 'date']).movieId.values.astype(str).tolist())

    word2vec = gensim.models.Word2Vec(movie_sequences, size=embedding_size, window=50000000, min_count=1, iter=word2vec_iter)
    word2vec.save("word2vec_" + str(embedding_size) + ".bin")

# STEP 2: CREATE TRAINING SAMPLES USING THE EMBEDDINGS

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

    # Obtain movies_ids and their embeddings
    user_movie_ids_train = group[group['date'] < date_test].sort_values('date').movieId.values.tolist()
    user_movie_ids_train_str = group[group['date'] < date_test].sort_values('date').movieId.values.astype(str).tolist()
    num_movies_train = len(user_movie_ids_train)
    embeddings_train = [word2vec[x] for x in user_movie_ids_train_str]
    embeddings_train = np.array(embeddings_train).reshape((len(embeddings_train), embedding_size))

    # Only add users with min_seq_length movies rated
    if num_movies_train > min_seq_length:

        # Create training sample
        for j in range(min_seq_length, num_movies_train):
            x_train = np.zeros((max_seq_length, embedding_size), dtype=np.float32)
            y_train = np.zeros(num_diff_items, dtype=np.int8)
            start = max(0, (j - max_seq_length))
            x_train[0:(j - start), :] = embeddings_train[start:j, :]
            y_movie_id = user_movie_ids_train[j]
            y_train[movieIds[y_movie_id]] = 1

            X_train.append(x_train)
            Y_train.append(sparse.csr_matrix(y_train, dtype=dtype_sparse))
            user_train.append(name)

        # Create sample for test
        x_test = np.zeros((max_seq_length, embedding_size), dtype=np.float32)
        num_movies_sample = min(max_seq_length, num_movies_train)
        x_test[0:num_movies_sample, :] = embeddings_train[-num_movies_sample:, :]

        user_movies_ids_test = group[group['date'] >= date_test].sort_values('date').movieId.values
        one_hot_pos_test = movieIds[user_movies_ids_test]
        num_movies_test = len(one_hot_pos_test)

        if num_movies_test > 0:
            user_test.append(name)
            time_idx = np.array(range(0, num_movies_test))
            y_test = np.zeros((num_movies_test, num_diff_items), dtype=np.int8)
            y_test[time_idx, one_hot_pos_test] = 1

            X_test.append(x_test)
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

# STEP 3: Save pickles
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
with open("pickles/movielens/X_test_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("pickles/movielens/Y_test_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
X_test = []
Y_test = []
gc.collect()
with open("pickles/movielens/Y_train_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
Y_train = []
gc.collect()
with open("pickles/movielens/X_train_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_2009_filter20.pickle", 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Saved pickles!')


