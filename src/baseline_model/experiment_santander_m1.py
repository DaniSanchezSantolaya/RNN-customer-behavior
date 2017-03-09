import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import pickle
import os
from load_santander import load_train_csv, load_test_csv
import tensorflow as tf
from tensorflow.contrib import rnn
import sys


random.seed(17)
np.random.seed(17)


#python experiment_santander_m1.py load_pickles p_val seq_length opt learning_rate n_hidden batch_size rnn_type rnn_layers dropout l2_reg type_output max_steps
#python experiment_santander_m1.py True 0.20 17 adam 0.0001 64 128 lstm 1 0.1 output_type 50000
#C:\Projects\Thesis\src\baseline_model>python experiment_santander_m3.py True 0.20 17 adam 0.0001 64 128 lstm2 2 0 0 sigmoid 500000

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


b_load_pickles = str2bool(sys.argv[1])
p_val = float(sys.argv[2])
seq_length = int(sys.argv[3])

opt = sys.argv[4]
learning_rate = float(sys.argv[5])
n_hidden = int(sys.argv[6])
batch_size = int(sys.argv[7])
rnn_type = sys.argv[8]
rnn_layers = int(sys.argv[9])
dropout = float(sys.argv[10])
l2_reg = float(sys.argv[11])
type_output = sys.argv[12]
max_steps = int(sys.argv[13])

name_submission = 'kaggle_submissions/m1-' + 'interactions_' + str(seq_length) + '-' + opt + '-lrate_' + str(learning_rate) + '-hidden_' + str(n_hidden) + '-bSize_' + str(batch_size) + '-' + rnn_type + '-rnn_layers_' + str(rnn_layers) + '-dropout_' + str(dropout) + '-L2_' + str(l2_reg) + '-typeoutput_' + str(type_output) + '-max_steps_' + str(max_steps) + '.csv'

print('Arguments: ')
print('Load pickle: ' + str(b_load_pickles))
print('p_val: ' + str(p_val))
print('seq_length: ' + str(seq_length))
print('opt: ' + str(opt))
print('learning_rate: ' + str(learning_rate))
print('n_hidden: ' + str(n_hidden))
print('batch_size: ' + str(batch_size))
print('rnn_type: ' + str(rnn_type))
print('rnn_layers: ' + str(rnn_layers))
print('dropout: ' + str(dropout))
print('l2_reg: ' + str(l2_reg))
print('type_output: ' + str(type_output))
print('max_steps: ' + str(max_steps))

	


	
def build_train_and_test():


    #Build Training Matrix
    X_train = []
    Y_train = []
    grouped = df.groupby('ncodpers')
    prob_discard = 0.75
    val_date = df['fecha_dato'].unique()[-1]
    a = 0
    for ncodpers,group in grouped:
        #if ncodpers == 1048484:
        for i in range(len(group)):
            if (i-seq_length) >= 0:
                #print('train in index ' + str(i) + ' fecha ' + str(group.iloc[i]['fecha_dato']))
                x = group[target_columns].values[(i-seq_length):i,:]
                y = group[target_columns].values[i,:]
                if False in (x[-1] == y): #if the products doesn't change in last time step, save train sample, if not discard it with some probaiblity 
                    X_train.append(x)
                    Y_train.append(y)
                elif np.random.rand() > prob_discard:
                    X_train.append(x)
                    Y_train.append(y)                   
            #else:
                #print('Not enough data points past in index ' + str(i) + ' fecha ' + str(group.iloc[i]['fecha_dato']))
        a = a + 1
        if a % 10000 == 0:
            print(a)
        if a % 350000 == 0:
            break
		
    diff_idx = []
    same_idx = []
    for i in range(len(X_train)):
        if False in (X_train[i][-1] == Y_train[i]):
            diff_idx.append(i)
        else:
            same_idx.append(i)
    print('Fnal train size: X' + str(len(X_train)))
    print('Fnal train size: Y' + str(len(Y_train)))
    print('same products: ' + str(len(same_idx)))
    print('Samples with changes in the last timestamp: ' + str(len(diff_idx)))
    
        
    #Build test matrix
    ncodpers_test = df_test['ncodpers']
    i = 0
    print(len(ncodpers_test))
    X_test = np.zeros((len(ncodpers_test), seq_length, len(target_columns)), dtype=np.float16)
    for ncodpers in ncodpers_test:
        #if ncodpers == 1048484:
        x = df[df['ncodpers']== ncodpers][target_columns].values[(-seq_length):]
        if len(x) < seq_length:
            result = np.zeros((seq_length, len(target_columns)))
            result[(seq_length - len(x)):,:] = x
            x = result
        #X_test.append(x)
        X_test[i] = x
        i = i + 1
        if i % 5000 == 0:
            print(i)

            

        
    #Save pickle
    with open('pickles/X_train_m1_' + str(seq_length) + '.pickle', 'wb') as handle:
        pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/X_test_m1_' + str(seq_length) + '.pickle', 'wb') as handle:
        pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('pickles/Y_train_m1_' + str(seq_length) + '.pickle', 'wb') as handle:
        pickle.dump(Y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X_train, X_test, Y_train
	 
	
#### Load pickle
def load_pickles():
	with open('pickles/X_train_m1_' + str(seq_length) + '.pickle', 'rb') as handle:
		X_train = pickle.load(handle)
	with open('pickles/X_test_m1_' + str(seq_length) + '.pickle', 'rb') as handle:
		X_test = pickle.load(handle)
	with open('pickles/Y_train_m1_' + str(seq_length) + '.pickle', 'rb') as handle:
		Y_train = pickle.load(handle)

	return X_train, X_test, Y_train
	
def transform_to_numpy(X_train, Y_train, X_test):
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    #Fix nulls comming from the original training set
    X_train[np.isnan(X_train)] = 0
    Y_train[np.isnan(Y_train)] = 0
    X_test[np.isnan(X_test)] = 0

    #Change negative interactions to 0 in the target, as we are only trying to predict adds
    Y_train[Y_train == -1] = 0

    return X_train, Y_train, X_test
	
def generate_validation_set(X_train, Y_train, X_test):
	print('Initial X_train size: ' + str( len(X_train)))
	print('Initial Y_train size: ' + str( len(Y_train)))
	print('Initial X_test size: ' + str( len(X_test)))

	num_val = int(len(X_train) * p_val)

	indices = np.random.permutation(X_train.shape[0])
	val_idx, training_idx = indices[:num_val], indices[num_val:]
	X_val, X_train = X_train[val_idx,:], X_train[training_idx,:]
	Y_val, Y_train  = Y_train[val_idx,:], Y_train[training_idx,:]
	
	print('Final X_train size: ' + str( len(X_train)))
	print('Final Y_train size: ' + str( len(Y_train)))
	print('Final X_val size: ' + str( len(X_val)))
	print('Final Y_val size: ' + str( len(Y_val)))
	print('Final X_test size: ' + str( len(X_test)))
	
	return X_train, X_val, X_test, Y_train, Y_val


class DataSet():
    """
    Utility class to handle dataset structure.
    """

    def __init__(self, X_train, Y_train):

        assert len(X_train) == len(Y_train), (
              "images.shape: {0}, labels.shape: {1}".format(str(len(X_train)), str(len(Y_train))))

        self._num_examples = len(X_train)
        self._X_train = X_train
        self._Y_train = Y_train
        self._epochs_completed = 0
        self._index_in_epoch = 0


    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            print('epochs completed: ' + str(self._epochs_completed))
            self._epochs_completed += 1

            #perm = np.arange(self._num_examples)
            #np.random.shuffle(perm)
            #self._X_train = self._X_train[perm]
            #self._Y_train = self._Y_train[perm]

            start = 0
            self._index_in_epoch = batch_size
            print(self._index_in_epoch)
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._X_train[start:end], self._Y_train[start:end]
	
	
df_test = load_test_csv()
if b_load_pickles:
    print('Load pickles')
    X_train, X_test, Y_train= load_pickles()
else:
    print('Build pickles')
    df = load_train_csv()
    X_train, X_test, Y_train = build_train_and_test()


print(len(X_train))
print(len(Y_train))
X_train, Y_train, X_test = transform_to_numpy(X_train, Y_train, X_test)
X_train, X_val, X_test, Y_train, Y_val = generate_validation_set(X_train, Y_train, X_test)
n_products = Y_train[0].shape[0]
ds = DataSet(X_train, Y_train)
X_train = []
Y_train = []


'''
TENSORFLOW MODEL
'''

n_input = n_products # data input
n_output = n_products
n_classes = 2 # MNIST total classes (0-9 digits)
seq_length = ds._X_train.shape[1]

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    
    x = tf.transpose(x, [1, 0, 2])
    
    
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    

    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    #x = tf.split(x, n_steps, 0)
    x = tf.split(split_dim=0, num_split=seq_length, value=x)


    # Define a lstm cell with tensorflow
    if rnn_type.lower() == 'lstm':
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    elif rnn_type.lower() == 'gru':
        rnn_cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    elif rnn_type == 'lstm2':
        rnn_cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    elif rnn_type == 'rnn':
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    elif rnn_type == 'lstm3':
        rnn_cell = tf.nn.rnn_cell.LayerNormBasicLSTMCell(n_hidden, dropout_keep_prob=0.9, dropout_prob_seed=17)
    elif rnn_type == 'lstm4':
        rnn_cell = tf.nn.rnn_cell.CoupledInputForgetGateLSTMCell(n_hidden, use_peepholes=False) #https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/CoupledInputForgetGateLSTMCell
    
    #Add dropout
    if dropout > 0:
        keep_prob = 1 - dropout
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    
    #stack rnn cells
    if rnn_layers > 1:
        #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_placeholder)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * rnn_layers)
    
    #TODO: Residual Wrappers: https://www.tensorflow.org/versions/master/api_docs/python/contrib.rnn/core_rnn_cell_wrappers__rnncells_that_wrap_other_rnncells_#ResidualWrapper
    
    
    # Get lstm cell output
    #tf.nn.rnn(cell, inputs,#initial_state=self._initial_state)
    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = tf.nn.rnn(rnn_cell, x, dtype=tf.float32)


    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def create_model():
    tf.reset_default_graph()

    # tf Graph input
    x = tf.placeholder("float", [None, seq_length, n_input], name='x')
    y = tf.placeholder("float", [None, n_output], name='y')

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_output]))
    }

    pred = RNN(x, weights, biases)

    # Define loss 
    if type_output.lower() == 'sigmoid':
        pred_prob = tf.sigmoid(pred)
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
    else:
        pred_prob = tf.nn.softmax(pred)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    
    #Add L2 regularizatoin loss
    reg_loss = l2_reg*tf.nn.l2_loss(weights['out'])
    cost = cost + reg_loss
    
    #Define optimizer
    if opt.lower() == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'adagraddao':
        optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'proximalgd':
        optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'proximaladagrad':
        optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(cost)
    elif opt.lower() == 'rms':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
        
        
    # Accuracy (Find a better evaluation)
    class_predictions = tf.cast(pred_prob > 0.5, tf.float32)
    correct_pred = tf.equal(class_predictions, y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #MAP
    num_samples = tf.shape(y)[0]
    cut_at_k = 7
    mask_vector = np.zeros(n_products)
    mask_vector[:cut_at_k] = 1
    sorted_pred_ind = tf.nn.top_k(pred_prob, n_products, sorted=True)[1]
    shape_ind = tf.shape(sorted_pred_ind)
    auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape_ind[:(sorted_pred_ind.get_shape().ndims - 1)]) + [n_products])], indexing='ij')
    t_sort_prod = tf.gather_nd(y, tf.stack(auxiliary_indices[:-1] + [sorted_pred_ind], axis=-1))

    num_added = tf.reduce_sum(t_sort_prod, axis=1)
    cumsum = tf.cumsum(t_sort_prod, axis=1)
    #repeat = tf.constant(np.repeat( (np.arange(n_products) + 1).reshape(1, n_products), y, axis=0))
    repeat = tf.reshape( (tf.range(n_products) + 1), [1, -1])
    repeat = tf.tile(repeat, [tf.shape(y)[0], 1])
    p_at_k = tf.cast(cumsum, tf.float32)/tf.cast(repeat, tf.float32)

    mask_at_k = tf.reshape(mask_vector, [1, -1])
    mask_at_k = tf.tile(mask_at_k, [tf.shape(y)[0], 1])
    #mask_at_k = tf.constant(np.repeat(mask_vector.reshape(1, n_products), num_samples, axis=0))


    sum_p_at_k = tf.reduce_sum((tf.cast(p_at_k, tf.float32) * tf.cast(mask_at_k, tf.float32)),  axis=1)
    t_cut = tf.fill((1, num_samples), cut_at_k)
    num_added_cut_k = tf.minimum(tf.cast(t_cut, tf.float32), tf.cast(num_added, tf.float32))
    num_added_cut_k = tf.maximum(num_added_cut_k, tf.cast(tf.fill((1, num_samples), 1), tf.float32))
    #AP = tf.cast(sum_p_at_k, tf.float32)/tf.cast(num_added_cut_k, tf.float32)
    AP = tf.cast(sum_p_at_k, tf.float32)/tf.cast(cut_at_k, tf.float32)
    MAP = tf.reduce_mean(AP)
    
    
    
    
    # Initializing the variables
    init = tf.global_variables_initializer()

    return pred, pred_prob, cost, optimizer, init, x, y, accuracy, MAP, AP
	
pred, pred_prob, cost, optimizer, init, x, y, accuracy, MAP, AP = create_model()

#Optimize
display_step = 100
checkpoint_freq_step = 100
#max_steps = 30000000
#max_steps = 600000
# Create a saver.
saver = tf.train.Saver()
checkpoint_dir = './checkpoints'
best_loss = 150000000
# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < max_steps:
    #while ds._epochs_completed < 10:
        batch_x, batch_y = ds.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        _, c = sess.run([optimizer, cost],
                                     feed_dict={x: batch_x, y: batch_y})
        
        if step % display_step == 0:
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch train Loss= " + 
                  "{:.6f}".format(loss))
            # Calculate val loss
            val_loss, val_acc, val_map, ap = sess.run([cost, accuracy, MAP, AP], feed_dict={x:X_val, y:Y_val})
            print('Val loss: ' + str(val_loss) + ', Val accuracy: ' + str(val_acc) + ', Val MAP: ' + str(val_map))
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'model_best.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
				#TODO save best path to load it later
                best_model = 'model_best.ckpt-' + str(step)
            
        if (step % checkpoint_freq_step == 0) or (step == max_steps - 1):
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            
        step += 1
    print("Optimization Finished!")
    
    pred_val = sess.run(pred_prob, feed_dict={x: X_val})
    # Calculate test loss
    val_loss, val_acc, val_map = sess.run([cost, accuracy, MAP], feed_dict={x: X_val, y: Y_val})
    print('Final validation loss: ' + str(val_loss) + ', Validation accuracy: ' + str(val_acc) + ', Validation MAP: ' + str(val_map))
	
	#Make predictions for test set with the best model
    if best_loss > val_loss:
        print('load best model')
        checkpoint_path = os.path.join(checkpoint_dir, best_model)
        saver.restore(sess, checkpoint_path)
    

    pred_test = []
    num_test_splits = 5 #Split to fit in memory when using large network architectures
    test_split_size = int(len(X_test)/num_test_splits)
    for i in range(num_test_splits):
        initial_idx = i * test_split_size
        final_idx = (i+1) * test_split_size
        print('Initial index: ' + str(initial_idx))
        print('final_idx index: ' + str(final_idx))
        if i < (num_test_splits - 1):
            pred_test_split = sess.run(pred_prob, feed_dict={x: X_test[initial_idx:final_idx]})
        else:
            pred_test_split = sess.run(pred_prob, feed_dict={x: X_test[initial_idx:]})
        pred_test.append(pred_test_split)
    #pred_test = sess.run(pred_prob, feed_dict={x: X_test})

pred_test = np.array(pred_test)
pred_test = pred_test.reshape((len(X_test), n_products))


ds._index_in_epoch = 0
ds._epochs_completed = 0



#Prediction only added but until seven items 
ncodpers_test = df_test['ncodpers']



f = open(name_submission, 'w')
f.write('ncodpers,added_products\n')
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

f.close() 

