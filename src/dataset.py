from scipy import sparse
import numpy as np
import pickle

class DataSet():
    """
    Utility class to handle dataset structure.
    """

    def __init__(self, X_train, Y_train, X_val, Y_val, X_train_last_month, Y_train_last_month, representation, X_local_test, Y_local_test, name_dataset = 'Santander'):

        print(name_dataset)
    
        assert len(X_train) == len(Y_train), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_train)), str(len(Y_train))))
        assert len(X_val) == len(Y_val), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_val)), str(len(Y_val))))
        assert len(X_train_last_month) == len(Y_train_last_month), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_train_last_month)), str(len(Y_train_last_month))))
              
        n_output = Y_train[0].toarray().shape[1]
        self._num_examples = len(X_train)
        self._X_train = X_train
        self._Y_train = Y_train
        self._X_val = X_val
        self._Y_val = Y_val
        # Transform X_train_last_month to np array
        self._X_train_last_month = []
        self._Y_train_last_month = []
        for x,y in zip(X_train_last_month, Y_train_last_month):
            self._X_train_last_month.append(x.toarray())
            self._Y_train_last_month.append(y.toarray().reshape(n_output))     
        self._X_train_last_month = np.array(self._X_train_last_month)
        self._Y_train_last_month = np.array(self._Y_train_last_month)
        # Transform X_local_test to np.array
        self._X_local_test = []
        self._Y_local_test = []
        for x,y in zip(X_local_test, Y_local_test):
            self._X_local_test.append(x.toarray())
            self._Y_local_test.append(y.toarray().reshape(n_output))     
        self._X_local_test = np.array(self._X_local_test)
        self._Y_local_test = np.array(self._Y_local_test)
        
        # Shuffle the train set
        perm = np.arange(self._num_examples)
        with open("pickles/movielens/train_perm.pickle", 'wb') as handle:
            pickle.dump(perm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.random.shuffle(perm)
        self._X_train = self._X_train[perm]
        self._Y_train = self._Y_train[perm]
        #Shuffle the validation set
        perm = np.arange(len(self._X_val))
        with open("pickles/movielens/val_perm.pickle", 'wb') as handle:
            pickle.dump(perm, handle, protocol=pickle.HIGHEST_PROTOCOL)
        np.random.shuffle(perm)
        self._X_val = self._X_val[perm]
        self._Y_val = self._Y_val[perm]
        
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._representation = representation
        self._name_dataset = name_dataset
        
        self._index_in_epoch_val = 0


    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            print('epochs completed: ' + str(self._epochs_completed))
            self._epochs_completed += 1

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X_train = self._X_train[perm]
            self._Y_train = self._Y_train[perm]

            start = 0
            self._index_in_epoch = batch_size
            print(self._index_in_epoch)
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        X = self._X_train[start:end]
        Y = self._Y_train[start:end]
        #seq_length = self._X_train[0].toarray().shape[0]
        #n_features = self._X_train[0].toarray().shape[1]
        n_output = self._Y_train[0].toarray().shape[1]
        batch_x = []
        batch_y = []
        for x,y in zip(X,Y):
            batch_x.append(x.toarray())
            batch_y.append(y.toarray().reshape(n_output))
        return np.array(batch_x), np.array(batch_y)
        
    def next_batch_val(self, batch_size):
        start = self._index_in_epoch_val
        self._index_in_epoch_val += batch_size
        if self._index_in_epoch_val > self._num_examples:

            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._X_val = self._X_val[perm]
            self._Y_val = self._Y_val[perm]

            start = 0
            self._index_in_epoch_val = batch_size
            print(self._index_in_epoch_val)
            assert batch_size <= self._num_examples

        end = self._index_in_epoch_val
        X = self._X_val[start:end]
        Y = self._Y_val[start:end]

        n_output = self._Y_val[0].toarray().shape[1]
        batch_x = []
        batch_y = []
        for x,y in zip(X,Y):
            batch_x.append(x.toarray())
            batch_y.append(y.toarray().reshape(n_output))
        return np.array(batch_x), np.array(batch_y)