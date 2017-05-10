
import tensorflow as tf
import numpy as np
import os
#from PhasedLSTMCell_v1 import *
#from PhasedLSTMCell import *
import time
import sys
import pickle

#Define RNN namespace according to tensorflow version
if tf.__version__ == '0.12.0':
    rnn_namespace = tf.nn.rnn_cell
elif tf.__version__ == '1.0.1':
    rnn_namespace = tf.contrib.rnn

b_val_in_batches = True

b_compute_acc = False
b_compute_map = False
    
def _seq_length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

def _last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)

    return relevant
    
    
    
class RNN_dynamic:

    def __init__(self, parameters):

        self.parameters = parameters
        
    def create_model(self):
        tf.reset_default_graph()

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.parameters['seq_length'], self.parameters['n_input']], name='x')
        self.y = tf.placeholder("float", [None, self.parameters['n_output']], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.parameters['n_hidden'], self.parameters['n_output']], stddev=self.parameters['init_stdev']), name='w_out')
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.parameters['n_output']]), name='b_out')
        }
        if self.parameters['embedding_size'] > 0:
            self.weights['emb'] = tf.Variable(tf.random_normal([self.parameters['n_input'], self.parameters['embedding_size']], stddev=self.parameters['init_stdev']), name='w_emb')
        

        # Define a lstm cell with tensorflow
        if self.parameters['rnn_type'].lower() == 'lstm':
            rnn_cell = rnn_namespace.BasicLSTMCell(self.parameters['n_hidden'], forget_bias=1.0)
        elif self.parameters['rnn_type'].lower() == 'gru':
            rnn_cell = rnn_namespace.GRUCell(self.parameters['n_hidden'])
        elif self.parameters['rnn_type'] == 'lstm2':
            rnn_cell = rnn_namespace.LSTMCell(self.parameters['n_hidden'])
        elif self.parameters['rnn_type'] == 'rnn':
            rnn_cell = rnn_namespace.BasicRNNCell(self.parameters['n_hidden'])
        elif self.parameters['rnn_type'] == 'plstm':
            rnn_cell = PhasedLSTMCell(self.parameters['n_hidden'], use_peepholes=True, state_is_tuple=True)

                
        #Add dropout
        if self.parameters['dropout'] > 0:
            rnn_cell = rnn_namespace.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)
            #keep_prob = 1 - self.parameters['dropout']
            #rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
            
            
        if self.parameters['rnn_layers'] > 1:
            rnn_cell = rnn_namespace.MultiRNNCell([rnn_cell] * self.parameters['rnn_layers'])  
            
        
        if self.parameters['embedding_size'] > 0:   
            self.x_reshaped = tf.reshape(self.x, [-1, int(self.x.get_shape()[2])])
            v = tf.matmul(self.x_reshaped, self.weights['emb'])
            v_reshaped = tf.reshape(v, [-1, self.parameters['seq_length'], self.parameters['embedding_size']])
            #if self.parameters['dropout'] > 0:
                #tf.nn.dropout(v_reshaped, (1 - self.dropout_rate)) # Check
            tf.summary.histogram('embeddings', v_reshaped)
            outputs, states = tf.nn.dynamic_rnn(
                rnn_cell,
                v_reshaped,
                dtype=tf.float32,
                sequence_length=_seq_length(v_reshaped)
            )
        else:
            outputs, states = tf.nn.dynamic_rnn(
                rnn_cell,
                self.x,
                dtype=tf.float32,
                sequence_length=_seq_length(self.x)
            )

        #Obtaining the correct output state
        if self.parameters['padding'].lower() == 'right': #If padding zeros is at right, we need to get the right output, since the last is not validation
            self.last_relevant_output = _last_relevant(outputs, _seq_length(self.x))
        elif self.parameters['padding'].lower() == 'left':
            self.last_relevant_output = outputs[:,-1,:]
        tf.summary.histogram('last_relevant_output', self.last_relevant_output)

        
        #self._debug = logits
        #self._debug = outputs

        if self.parameters['type_output'].lower() == 'sigmoid':
            self.logits = tf.matmul(self.last_relevant_output, self.weights['out']) + self.biases['out']
            self.pred_prob = tf.sigmoid(self.logits)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y))
        elif self.parameters['type_output'].lower() == 'softmax':
            self.logits = tf.matmul(self.last_relevant_output, self.weights['out']) + self.biases['out']
            self.pred_prob = tf.nn.softmax(self.logits)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            tf.summary.histogram('logits', self.logits)
            tf.summary.histogram('pred_prob', self.pred_prob)
        elif self.parameters['type_output'].lower() == 'sampled_softmax':
            self.logits = tf.matmul(self.last_relevant_output, self.weights['out']) + self.biases['out']
            self.pred_prob = tf.nn.softmax(self.logits)
            weights_out_t = tf.transpose(self.weights['out'])
            labels_ids = tf.where(tf.equal(self.y, 1))[:, 1]
            labels_t = tf.reshape(labels_ids, [-1, 1])
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=weights_out_t, biases=self.biases['out'], inputs=self.last_relevant_output,
                               labels=labels_t, num_sampled=128, num_classes=self.parameters['n_output']))
            


        #Add L2 regularizatoin loss
        self.loss = self.loss + self.parameters['l2_reg'] * tf.nn.l2_loss(self.weights['out'])
        if self.parameters['embedding_size'] > 0:
            self.loss += self.parameters['l2_reg'] * tf.nn.l2_loss(self.weights['emb'])
        for v in tf.trainable_variables():
            if (v.name == 'RNN/GRUCell/Gates/Linear/Matrix:0') or (v.name == 'rnn/gru_cell/gates/weights:0'):
                gru_w1 = v
            elif (v.name == 'RNN/GRUCell/Gates/Linear/Bias:0') or (v.name == 'rnn/gru_cell/gates/biases:0'):
                gru_b1 = v
            elif (v.name == 'RNN/GRUCell/Candidate/Linear/Matrix:0') or (v.name == 'rnn/gru_cell/candidate/weights:0'):
                gru_w2 = v
            elif (v.name == 'RNN/GRUCell/Candidate/Linear/Bias:0') or (v.name == 'rnn/gru_cell/candidate/biases:0'):
                gru_b2 = v
            elif (v.name == 'RNN/BasicLSTMCell/Linear/Matrix:0') or (v.name == 'rnn/basic_lstm_cell/weights:0'):
                basic_lstm_w = v
            elif (v.name == 'RNN/BasicLSTMCell/Linear/Bias:0') or (v.name == 'rnn/basic_lstm_cell/biases:0'):
                basic_lstm_b = v
            elif (v.name == 'RNN/LSTMCell/W_0:0') or (v.name == 'rnn/lstm_cell/weights:0'):
                lstm_w = v
            elif (v.name == 'RNN/LSTMCell/B:0') or (v.name == 'rnn/lstm_cell/biases:0'):
                lstm_b = v

        if self.parameters['rnn_type'].lower() == 'gru':
            self.loss += self.parameters['l2_reg'] * tf.nn.l2_loss(gru_w1) + self.parameters['l2_reg'] * tf.nn.l2_loss(gru_w2)
            tf.summary.histogram('gru_w1', gru_w1)
            tf.summary.histogram('gru_b1', gru_b1)
            tf.summary.histogram('gru_w2', gru_w2)
            tf.summary.histogram('gru_b2', gru_b2)
        elif self.parameters['rnn_type'].lower() == 'lstm':
            self.loss += self.parameters['l2_reg'] * tf.nn.l2_loss(basic_lstm_w)
            tf.summary.histogram('basic_lstm_w', basic_lstm_w)
            tf.summary.histogram('basic_lstm_b', basic_lstm_b)
        elif self.parameters['rnn_type'].lower() == 'lstm2':
            self.loss += self.parameters['l2_reg'] * tf.nn.l2_loss(lstm_w)
            tf.summary.histogram('lstm_w', lstm_w)
            tf.summary.histogram('lstm_b', lstm_b)

        #Define optimizer
        if self.parameters['opt'].lower() == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'adagraddao':
            self.optimizer = tf.train.AdagradDAOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'ftrl':
            self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'proximalgd':
            self.optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'proximaladagrad':
            self.optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        elif self.parameters['opt'].lower() == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.parameters['learning_rate']).minimize(self.loss)
        

        if b_compute_acc:
            class_predictions = tf.cast(self.pred_prob > 0.5, tf.float32)
            correct_pred = tf.equal(class_predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
        else:
            self.accuracy = tf.constant(0)

        if b_compute_map:
            #MAP
            num_samples = tf.shape(self.y)[0]
            cut_at_k = 7
            mask_vector = np.zeros(self.parameters['n_output'])
            mask_vector[:cut_at_k] = 1
            sorted_pred_ind = tf.nn.top_k(self.pred_prob, self.parameters['n_output'], sorted=True)[1]
            shape_ind = tf.shape(sorted_pred_ind)
            auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape_ind[:(sorted_pred_ind.get_shape().ndims - 1)]) + [self.parameters['n_output']])], indexing='ij')
            t_sort_prod = tf.gather_nd(self.y, tf.stack(auxiliary_indices[:-1] + [sorted_pred_ind], axis=-1))

            num_added = tf.reduce_sum(t_sort_prod, axis=1)
            cumsum = tf.cumsum(t_sort_prod, axis=1)
            #repeat = tf.constant(np.repeat( (np.arange(self.parameters['n_output']) + 1).reshape(1, self.parameters['n_output']), y, axis=0))
            repeat = tf.reshape( (tf.range(self.parameters['n_output']) + 1), [1, -1])
            repeat = tf.tile(repeat, [tf.shape(self.y)[0], 1])
            p_at_k = tf.cast(cumsum, tf.float32)/tf.cast(repeat, tf.float32)

            mask_at_k = tf.reshape(mask_vector, [1, -1])
            mask_at_k = tf.tile(mask_at_k, [tf.shape(self.y)[0], 1])
            #mask_at_k = tf.constant(np.repeat(mask_vector.reshape(1, self.parameters['n_output']), num_samples, axis=0))


            sum_p_at_k = tf.reduce_sum((tf.cast(p_at_k, tf.float32) * tf.cast(mask_at_k, tf.float32)),  axis=1)
            t_cut = tf.fill((1, num_samples), cut_at_k)
            num_added_cut_k = tf.minimum(tf.cast(t_cut, tf.float32), tf.cast(num_added, tf.float32))
            num_added_cut_k = tf.maximum(num_added_cut_k, tf.cast(tf.fill((1, num_samples), 1), tf.float32))
            # AP = tf.cast(sum_p_at_k, tf.float32)/tf.cast(num_added_cut_k, tf.float32)
            AP = tf.cast(sum_p_at_k, tf.float32)/tf.cast(cut_at_k, tf.float32)
            self.MAP = tf.reduce_mean(AP)
            
            # Add summaries
            tf.summary.scalar('MAP', self.MAP)
        else:
            self.MAP = tf.constant(0)
            
        tf.summary.scalar('loss', self.loss)

        tf.summary.histogram('W_out', self.weights['out'])
        if self.parameters['embedding_size'] > 0:
            tf.summary.histogram('W_emb', self.weights['emb'])

               
        self.init = tf.global_variables_initializer()
        
    def train(self, ds):

        ###try:
        actual_epoch = 0
        train_loss_list = []
        val_loss_list = []
        dropout_keep_prob = 1 - self.parameters['dropout']
        display_step = 50
        checkpoint_freq_step = 50
        #max_steps = 30000000
        #max_steps = 600000
        # Create a saver.
        saver_last = tf.train.Saver()
        saver_best = tf.train.Saver()
        self.parameters_str = str('rep' + str(ds._representation) + '-' + self.parameters['rnn_type']) + '-' + str(self.parameters['n_hidden']) + '-' + str(self.parameters['rnn_layers']) + '-' + str(self.parameters['batch_size']) + '-' + str(self.parameters['opt']) + '-' + str(self.parameters['max_steps'])
        self.parameters_str += '-' + time.strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = './checkpoints/' + self.parameters_str
        if not tf.gfile.Exists(checkpoint_dir):
            tf.gfile.MakeDirs(checkpoint_dir)
            tf.gfile.MakeDirs(checkpoint_dir + '/best_model')
            tf.gfile.MakeDirs(checkpoint_dir + '/last_model')
        self.best_loss = 150000000
        # Launch the graph
        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('tensorboard/' + ds._name_dataset + '/' + str(self.parameters_str) + '/train', sess.graph)
            train_last_month_writer = tf.summary.FileWriter('tensorboard/' + ds._name_dataset + '/' + str(self.parameters_str) + '/train_last_month', sess.graph)
            val_writer = tf.summary.FileWriter('tensorboard/' + ds._name_dataset + '/'  + str(self.parameters_str) + '/val', sess.graph)
            test_writer = tf.summary.FileWriter('tensorboard/' + ds._name_dataset + '/'  + str(self.parameters_str) + '/test', sess.graph)
            sess.run(self.init)

            step = 1

            for v in tf.trainable_variables():
                print(v.name)
            print('---------')
            sys.stdout.flush()

            #weight = [v for v in tf.global_variables() if v.name == "Variable_23:0"][0]
            # Keep training until reach max iterations
            while step * self.parameters['batch_size'] < self.parameters['max_steps']:
                total_iterations = step * self.parameters['batch_size']


                #Obtain batch for this iteration
                batch_x, batch_y, new_epoch = ds.next_batch(self.parameters['batch_size'])
                if new_epoch:
                    actual_epoch += 1

                # Run optimization op (backprop)
                #sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout_keep_prob: 1})
                _, c = sess.run([self.optimizer, self.loss],
                                             feed_dict={self.x: batch_x, self.y: batch_y, self.dropout_keep_prob: dropout_keep_prob})


                if (step % display_step == 0) or ((total_iterations + self.parameters['batch_size'])  >= (self.parameters['max_steps'] - 1)):
                    # Calculate batch loss
                    train_minibatch_loss, summary = sess.run([self.loss, merged], feed_dict={self.x: batch_x, self.y: batch_y, self.dropout_keep_prob: 1})
                    print("Iter " + str(total_iterations) + ", Minibatch train Loss= " +
                          "{:.6f}".format(train_minibatch_loss))
                    train_writer.add_summary(summary, total_iterations)
                    train_loss_list.append(train_minibatch_loss)
                    print("Iter "+ str(total_iterations) + ", mean last 25 train loss: " + str(np.mean(train_loss_list[-25:])))
                    # Calculate training loss at last month
                    if len(ds._X_train_last_month) > 0:
                        train_last_month_loss, train_last_month_accuracy, train_last_month_map, summary = sess.run([self.loss, self.accuracy, self.MAP, merged], feed_dict={self.x: ds._X_train_last_month, self.y: ds._Y_train_last_month, self.dropout_keep_prob: 1})
                        print("Iter " + str(total_iterations) + ", Last month train Loss= " +
                              "{:.6f}".format(train_last_month_loss) + ", Last month train Accuracy= " +
                              "{:.6f}".format(train_last_month_accuracy) + ", Last train Map= " +
                              "{:.6f}".format(train_last_month_map))
                        train_last_month_writer.add_summary(summary, total_iterations)
                    # Calculate val loss
                    if ds._name_dataset.lower() == 'santander':
                        self.val_loss, val_acc, val_map, summary = sess.run([self.loss, self.accuracy, self.MAP, merged], feed_dict={self.x:ds._X_val, self.y:ds._Y_val, self.dropout_keep_prob: 1})
                        val_writer.add_summary(summary, total_iterations)
                        print("Iter " + str(total_iterations) + ", Validation  Loss= " +
                              "{:.6f}".format(self.val_loss) + ", Validation  Accuracy= " +
                              "{:.6f}".format(val_acc) + ", Validation Map= " +
                              "{:.6f}".format(val_map))
                    elif ds._name_dataset.lower() == 'movielens':
                        if b_val_in_batches: # Do for only one batch
                            if False: #TODO FIX: is giving exception this block when new epoch
                                # Compute for the whole validation set (in batches)
                                val_loss_list = []
                                val_batch_size = 100
                                start = 0
                                end = start + val_batch_size
                                while end < len(ds._X_val):
                                    x_val_batch = [x.toarray() for x in ds._X_val[start:end]]
                                    y_val_batch = [y.toarray().reshape(y.toarray().shape[1]) for y in
                                                   ds._Y_val[start:end]]
                                    start = end
                                    end = start + val_batch_size
                                    val_loss = sess.run(
                                        [self.loss],feed_dict={self.x: x_val_batch, self.y: y_val_batch,
                                                   self.dropout_keep_prob: 1})
                                    val_loss_list.append(val_loss)
                                # Do the last batch
                                x_val_batch = [x.toarray() for x in ds._X_val[start:]]
                                y_val_batch = [y.toarray().reshape(y.toarray().shape[1]) for y in ds._Y_val[start:]]
                                val_loss = sess.run([self.loss], feed_dict={self.x: x_val_batch,
                                                                                 self.y: y_val_batch,
                                                                                 self.dropout_keep_prob: 1})
                                val_loss_list.append(val_loss)
                                # Compute mean of bathces val loss
                                self.val_loss = np.mean(val_loss_list)
                                val_map = np.mean(val_map)
                                print("Epoch " + str(actual_epoch) + ", Validation  Loss= " +
                                      "{:.6f}".format(self.val_loss))
                            else:
                                # Obtain val batch for this iteration
                                batch_x_val, batch_y_val = ds.next_batch_val(self.parameters['batch_size'])
                                self.val_loss, summary = sess.run([self.loss, merged], feed_dict={self.x:batch_x_val, self.y:batch_y_val, self.dropout_keep_prob: 1})
                                val_writer.add_summary(summary, total_iterations)
                                print("Iter " + str(total_iterations) + ", Validation  Loss= " +
                                      "{:.6f}".format(self.val_loss))
                                val_loss_list.append(self.val_loss)
                                print("Iter "+ str(total_iterations) + ", mean last 25 validation loss: " + str(np.mean(val_loss_list[-25:])))

                    # Calculate test loss
                    if len(ds._X_local_test) > 0:
                        self.test_loss, test_acc, test_map, summary = sess.run([self.loss, self.accuracy, self.MAP, merged], feed_dict={self.x:ds._X_local_test, self.y:ds._Y_local_test, self.dropout_keep_prob: 1})
                        test_writer.add_summary(summary, total_iterations)
                        print("Iter " + str(total_iterations) + ", Test Loss= " +
                              "{:.6f}".format(self.test_loss) + ", Test Accuracy= " +
                              "{:.6f}".format(test_acc) + ", Test Map= " +
                              "{:.6f}".format(test_map))


                    #If best loss save the model as best model so far
                    if b_val_in_batches and ds._name_dataset.lower() == 'movielens':
                        if np.mean(val_loss_list[-25:]) < self.best_loss:
                            self.best_loss = np.mean(val_loss_list[-25:])
                            checkpoint_dir_tmp = checkpoint_dir + '/best_model/'
                            checkpoint_path = os.path.join(checkpoint_dir_tmp, 'model_best.ckpt')
                            saver_best.save(sess, checkpoint_path, global_step=total_iterations)
                            self.best_model_path = 'model_best.ckpt-' + str(total_iterations)
                            #print('-->save best model: ' + str(checkpoint_path) + ' - step: ' + str(step) + ' best_model_path: ' + str(self.best_model_path))
                    else:
                        if self.val_loss < self.best_loss:
                            self.best_loss = self.val_loss
                            checkpoint_dir_tmp = checkpoint_dir + '/best_model/'
                            checkpoint_path = os.path.join(checkpoint_dir_tmp, 'model_best.ckpt')
                            saver_best.save(sess, checkpoint_path, global_step=total_iterations)
                            self.best_model_path = 'model_best.ckpt-' + str(total_iterations)

                    print('------------------------------------------------')
                    sys.stdout.flush()

                #Save check points periodically or in last iteration
                if (step % checkpoint_freq_step == 0) or ( (total_iterations + self.parameters['batch_size'])  >= (self.parameters['max_steps'] - 1)):
                    checkpoint_dir_tmp =  checkpoint_dir + '/last_model/'
                    checkpoint_path = os.path.join(checkpoint_dir_tmp, 'last_model.ckpt')
                    saver_last.save(sess, checkpoint_path, global_step=total_iterations)
                    self.last_model_path = 'last_model.ckpt-' + str(total_iterations)
                    #print('-->save checkpoint model: ' + str(checkpoint_path) + ' - step: ' + str(step) + ' last_model_path: ' + str(self.last_model_path))

                step += 1
            print("Optimization Finished!")


                #pred_val = sess.run(self.pred_prob, feed_dict={self.x: ds._X_val})
                # Calculate val loss
                #self.val_loss, val_acc, val_map = sess.run([self.loss, self.accuracy, self.MAP], feed_dict={self.x: ds._X_val, self.y: ds._Y_val})
                #print('Final validation loss: ' + str(self.val_loss) + ', Validation accuracy: ' + str(val_acc) + ', Validation MAP: ' + str(val_map))
        ###except Exception as e:
            ###print('Execption: ' + str(e))
            ###with open("batch_x.pickle",'wb') as handle:
                ###pickle.dump(batch_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
            ###with open("batch_y.pickle",'wb') as handle:
                ###pickle.dump(batch_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def predict(self, X_test, checkpoint_path = None, num_test_splits = 1):
        # Make predictions for test set with the best model
        if checkpoint_path is None:
            checkpoint_dir = './checkpoints/' + self.parameters_str
            # CHECK: is removing the best model sometimes
            if self.best_loss < self.val_loss:
                checkpoint_dir_tmp =  checkpoint_dir + '/best_model/'
                checkpoint_path = os.path.join(checkpoint_dir_tmp, self.best_model_path)
            else:
                checkpoint_dir_tmp =  checkpoint_dir + '/last_model/'
                checkpoint_path = os.path.join(checkpoint_dir_tmp, self.last_model_path)

        saver = tf.train.Saver()

        # checkpoint_path = os.path.join(checkpoint_dir, self.last_model_path)
        with tf.Session() as sess:
            print('load model: ' + str(checkpoint_path))
            saver.restore(sess, checkpoint_path)
            

            #pred_test = []
            pred_test = np.zeros((len(X_test), self.parameters['n_output']))
            #num_test_splits = 1 #Split to fit in memory when using large network architectures
            test_split_size = int(len(X_test)/num_test_splits)
            for i in range(num_test_splits):
                initial_idx = i * test_split_size
                final_idx = (i+1) * test_split_size
                #print('Initial index: ' + str(initial_idx))
                #print('final_idx index: ' + str(final_idx))
                if i < (num_test_splits - 1):
                    logits, pred_test_split = sess.run([self.logits, self.pred_prob], feed_dict={self.x: X_test[initial_idx:final_idx], self.dropout_keep_prob: 1})
                    #print(pred_test_split.shape)
                    pred_test[initial_idx:final_idx] = pred_test_split
                else:
                    logits, pred_test_split = sess.run([self.logits, self.pred_prob], feed_dict={self.x: X_test[initial_idx:], self.dropout_keep_prob: 1})
                    #print(pred_test_split.shape)
                    pred_test[initial_idx:] = pred_test_split
                #pred_test.append(pred_test_split)
                
            #pred_test = sess.run(self.pred_prob, feed_dict={x: X_test})

            #print(pred_test[0].shape)
            #pred_test = np.array(pred_test)
            #print(pred_test.shape)
            #pred_test = pred_test.reshape((len(X_test), self.parameters['n_output']))
        
        return logits, pred_test
        
    def get_last_hidden_state(self, X_test):
        #Make predictions for test set with the best model
        checkpoint_dir = './checkpoints'
        saver = tf.train.Saver()

        #CHECK: is removing the best model sometines
        if self.best_loss < self.val_loss:
            checkpoint_path = os.path.join(checkpoint_dir, self.best_model_path)
            print('load best model: ' + str(checkpoint_path))
        else:
            checkpoint_path = os.path.join(checkpoint_dir, self.last_model_path)
            print('load last model' + str(checkpoint_path))
        
        
        checkpoint_path = os.path.join(checkpoint_dir, self.last_model_path)
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            last_hidden_state = sess.run(self.last_relevant_output, feed_dict={self.x: X_test, self.dropout_keep_prob: 1})

            
        
        return last_hidden_state