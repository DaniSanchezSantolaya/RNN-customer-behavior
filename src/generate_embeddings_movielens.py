from gensim.models import word2vec
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
import gc
import sys
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

b_generate_training_samples = True
b_output_embeddings = True
load_pretrained_embeddings = True
embedding_size = 64
start_date_embeddings = '2009-01-01'
start_date_train = '2009-01-01'
date_test = '2014-10-01'
movies_min_ratings = 20
min_seq_length = 5
max_seq_length = 100
word2vec_iter = 25
window_size = 10
num_users_save_train = 500 # Save file every this number of users to avoid consume all the RAM
year = '2009'

# representation: 1: 1 sample per user, 2: data augmentation, 3: intermediate errors
representation = 2


last_part_filename = ""
if b_output_embeddings:
    last_part_filename = '_output'


# STEP 1: LOAD PRETRAINED EMBEDDINGS OR TRAIN THEM FROM SCRATCH

if load_pretrained_embeddings:
    word2vec = gensim.models.Word2Vec.load("word2vec_" + str(embedding_size) + ".bin")
    #word2vec =  gensim.models.KeyedVectors.load_word2vec_format("word2vec_" + str(embedding_size) + ".bin", binary=True)
    print('Loaded word2vec')
else:

    print('Train embeddings')

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

    word2vec = gensim.models.Word2Vec(movie_sequences, size=embedding_size, window=window_size, min_count=1, iter=word2vec_iter)
    word2vec.save("word2vec_" + str(embedding_size) + ".bin")

# STEP 2: CREATE TRAINING SAMPLES USING THE EMBEDDINGS
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

# Build array mapping movie_id --> position in one-hot encoding
print('Build array mapping movie_id --> position in one-hot encoding')
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
actual_num_file = 0
i = 0

# Save embedding matrix
if not load_pretrained_embeddings:
    W_emb = np.zeros((num_diff_items, embedding_size))
    for movieId in df_date['movieId'].unique():
        W_emb[movieIds[movieId], :] = word2vec[str(movieId)]
    with open("pickles/movielens/W_emb_64.pickle", 'wb') as handle:
        pickle.dump(W_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

if b_generate_training_samples:
    print('Generate training samples')
    for name, group in grouped:

        # Obtain movies_ids and their embeddings
        user_movie_ids_train = group[group['date'] < date_test].sort_values('date').movieId.values.tolist()
        user_movie_ids_train_str = group[group['date'] < date_test].sort_values('date').movieId.values.astype(str).tolist()
        one_hot_pos_train = movieIds[user_movie_ids_train]
        num_movies_train = len(user_movie_ids_train)
        embeddings_train = [word2vec[x] for x in user_movie_ids_train_str]
        embeddings_train = np.array(embeddings_train).reshape((len(embeddings_train), embedding_size))

        # Only add users with min_seq_length movies rated
        if num_movies_train > min_seq_length:
            ratings = np.zeros((num_movies_train, num_diff_items), dtype=np.int8)

            # Set 1 in the correspondent timestamp and one hot positions for every movie
            time_idx = np.array(range(0, num_movies_train))
            ratings[time_idx, one_hot_pos_train] = 1

            # Create training sample depending on the representation
            if representation == 2:
                for j in range(min_seq_length, num_movies_train):
                    x_train = np.zeros((max_seq_length, embedding_size), dtype=np.float32)
                    start = max(0, (j - max_seq_length))
                    x_train[0:(j - start), :] = embeddings_train[start:j, :]
                    X_train.append(x_train)
                    y_movie_id = user_movie_ids_train[j]
                    if not b_output_embeddings:
                        y_train = np.zeros(num_diff_items, dtype=np.int8)
                        y_train[movieIds[y_movie_id]] = 1
                        Y_train.append(sparse.csr_matrix(y_train, dtype=dtype_sparse))
                    else:
                        y_train = embeddings_train[j]
                        Y_train.append(y_train)
                    user_train.append(name)

            elif representation == 3:
                for i in range(0, num_movies_train, max_seq_length):
                    x_start = i
                    x_end = min(i + max_seq_length, num_movies_train - 1)
                    y_start = x_start + 1
                    y_end = x_end + 1
                    length_sample = x_end - x_start
                    if length_sample > min_seq_length:
                        x_train = np.zeros((max_seq_length, embedding_size), dtype=np.float32)
                        x_train[0:length_sample, :] = embeddings_train[x_start:x_end, :]
                        X_train.append(x_train)
                        if not b_output_embeddings:
                            y_train = np.zeros((max_seq_length, num_diff_items), dtype=np.int8)
                            y_train[0:length_sample, :] = ratings[y_start:y_end, :]
                            Y_train.append(sparse.csr_matrix(y_train, dtype=dtype_sparse))
                        else:
                            y_train = np.zeros((max_seq_length, embedding_size), dtype=np.float32)
                            y_train[0:length_sample, :] = embeddings_train[y_start:y_end, :]
                            Y_train.append(y_train)
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
        if i % num_users_save_train == 0:
            with open("pickles/movielens/Y_train_" + str(max_seq_length) + "_embeddings_" + str(
                    embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(actual_num_file) + last_part_filename + ".pickle", 'wb') as handle:
                pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            Y_train = []
            with open("pickles/movielens/X_train_" + str(max_seq_length) + "_embeddings_" + str(
                    embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(actual_num_file) + last_part_filename + ".pickle", 'wb') as handle:
                pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            X_train = []
            gc.collect()
            actual_num_file += 1
        if i % 2500 == 0:
            print(i)
            print('Num training samples: ' + str(len(user_train)))
            sys.stdout.flush()

    print('Num train samples: ' + str(len(X_train)))
    print('Num test samples: ' + str(len(X_test)))

    # STEP 3: Save pickles
    with open("pickles/movielens/discarded_users_" + str(max_seq_length) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'wb') as handle:
        pickle.dump(discarded_users, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("pickles/movielens/user_train_" + str(max_seq_length) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'wb') as handle:
        pickle.dump(user_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("pickles/movielens/user_test_" + str(max_seq_length) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'wb') as handle:
        pickle.dump(user_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df_date = []
    df = []
    df_2015 = []
    discarded_users = []
    user_train = []
    user_test = []
    gc.collect()
    with open("pickles/movielens/X_test_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("pickles/movielens/Y_test_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_" + year + "_filter20_rep" + str(representation) + ".pickle", 'wb') as handle:
        pickle.dump(Y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    X_test = []
    Y_test = []
    gc.collect()
    if len(Y_train) > 0:
        with open("pickles/movielens/Y_train_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(actual_num_file) + last_part_filename + ".pickle", 'wb') as handle:
            pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    Y_train = []
    gc.collect()
    if len(X_train) > 0:
        with open("pickles/movielens/X_train_" + str(max_seq_length) + "_embeddings_" + str(embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(actual_num_file) + last_part_filename + ".pickle", 'wb') as handle:
            pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved pickles!')


