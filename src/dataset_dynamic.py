from scipy import sparse
import numpy as np
import pickle


class DataSet():
    """
    Utility class to handle dataset structure.
    """


    def __init__(self, max_seq_length, embedding_size, year, representation, num_total_files, num_validation_file, b_output_embeddings, name_dataset='movielens'):

        assert num_validation_file != 0, ("Validation set file cannot be the first one")
        assert num_validation_file != num_total_files, ("Validation set file cannot be the last one")

        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.year = year
        self._representation = representation
        self.num_total_files = num_total_files
        self.num_validation_file = num_validation_file
        self._name_dataset = name_dataset

        self._actual_file = 0
        self._epochs_completed = 0
        self._index_in_file = 0
        self._index_in_epoch_val = 0

        if b_output_embeddings:
            self.last_part_filename = '_output'

        # Load first training file
        with open("pickles/movielens/X_train_" + str(max_seq_length) + "_embeddings_" + str(
                embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(self._actual_file) + self.last_part_filename + ".pickle", 'rb') as handle:
            self._X_train = pickle.load(handle)

        with open("pickles/movielens/Y_train_" + str(max_seq_length) + "_embeddings_" + str(
                embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(self._actual_file) + self.last_part_filename +  ".pickle", 'rb') as handle:
            self._Y_train = pickle.load(handle)

        self._num_samples_file = len(self._X_train)
        #print('Num samples file: ' + str(len(self._X_train)))

        # Load validation file
        with open("pickles/movielens/X_train_" + str(max_seq_length) + "_embeddings_" + str(
                embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(num_validation_file) + self.last_part_filename + ".pickle", 'rb') as handle:
            self._X_val = pickle.load(handle)
        with open("pickles/movielens/Y_train_" + str(max_seq_length) + "_embeddings_" + str(
                embedding_size) + "_" + year + "_filter20_rep" + str(representation) + "_file" + str(num_validation_file) + self.last_part_filename + ".pickle", 'rb') as handle:
            self._Y_val = pickle.load(handle)

        self._num_examples_val = len(self._X_val)

    def next_batch(self, batch_size):

        new_epoch = False
        start = self._index_in_file
        self._index_in_file += batch_size
        if self._index_in_file >= self._num_samples_file:
            #print('Training file ' + str(self._actual_file) + ' completed')
            # Change file
            self._actual_file += 1

            # If is the validation set, skip it
            if self._actual_file == self.num_validation_file:
                self._actual_file += 1
            # If we have gone through all files, start a new epoch from the first file
            if self._actual_file >= self.num_total_files:
                self._epochs_completed += 1
                #print('Epoch completed! epochs completed: ' + str(self._epochs_completed))
                self._actual_file = 0
                new_epoch = True

            #print('Start with training file: ' + str(self._actual_file))

            # Load next training file
            with open("pickles/movielens/X_train_" + str(self.max_seq_length) + "_embeddings_" + str(
                    self.embedding_size) + "_" + self.year + "_filter20_rep" + str(self._representation) + "_file" + str(
                self._actual_file) + self.last_part_filename + ".pickle", 'rb') as handle:
                self._X_train = pickle.load(handle)
            with open("pickles/movielens/Y_train_" + str(self.max_seq_length) + "_embeddings_" + str(
                    self.embedding_size) + "_" + self.year + "_filter20_rep" + str(self._representation) + "_file" + str(
                self._actual_file) + self.last_part_filename + ".pickle", 'rb') as handle:
                self._Y_train = pickle.load(handle)
            self._num_samples_file = len(self._X_train)
            #print('Num samples file: ' + str(len(self._X_train)))
            start = 0
            self._index_in_file = batch_size
            # print(self._index_in_file)
            assert batch_size <= self._num_samples_file

        end = self._index_in_file
        # START DEBUG
        if (self._actual_file == 3) and (self._index_in_file > 50232):
            print('start: ' + str(start))
            print('end: ' + str(end))
        # END DEBUG
        X = self._X_train[start:end]
        # if not numpy array is sparse matrix
        if not type(X[0]).__module__ == np.__name__:
            batch_x = []
            for x in X:
                batch_x.append(x.toarray())
            batch_x = np.array(batch_x)
        else:
            if type(X).__module__ == np.__name__:
                batch_x = X
            else:
                batch_x = np.array(X)
        Y = self._Y_train[start:end]
        if not type(Y[0]).__module__ == np.__name__:
            batch_y = []
            seq_length_y = self._Y_train[0].toarray().shape[0]
            n_output = self._Y_train[0].toarray().shape[1]
            for y in Y:
                if seq_length_y > 1:
                    batch_y.append(y.toarray())
                else:
                    batch_y.append(y.toarray().reshape(n_output))
            batch_y = np.array(batch_y)
        else:
            if type(Y).__module__ == np.__name__:
                batch_y = Y
            else:
                batch_y = np.array(Y)

        return batch_x, batch_y, new_epoch

    def next_batch_val(self, batch_size):
        start = self._index_in_epoch_val
        self._index_in_epoch_val += batch_size
        if self._index_in_epoch_val > self._num_examples_val:
            print('New val epoch')

            start = 0
            self._index_in_epoch_val = batch_size
            assert batch_size <= self._num_examples_val

        end = self._index_in_epoch_val
        X = self._X_val[start:end]
        Y = self._Y_val[start:end]
        # if not numpy array is sparse matrix
        if not type(self._X_val[0]).__module__ == np.__name__:
            batch_x = []
            for x in X:
                batch_x.append(x.toarray())
            batch_x = np.array(batch_x)
        else:
            if type(X).__module__ == np.__name__:
                batch_x = X
            else:
                batch_x = np.array(X)
        if not type(self._Y_val[0]).__module__ == np.__name__:
            n_output = self._Y_val[0].toarray().shape[1]
            batch_y = []
            seq_length_y = self._Y_val[0].toarray().shape[0]
            for y in Y:
                if seq_length_y > 1:
                    batch_y.append(y.toarray())
                else:
                    batch_y.append(y.toarray().reshape(n_output))
            batch_y = np.array(batch_y)
        else:
            if type(Y).__module__ == np.__name__:
                batch_y = Y
            else:
                batch_y = np.array(Y)
        return batch_x, batch_y

