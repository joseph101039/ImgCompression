from __future__ import print_function
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt     #Installation command: python -mpip install -U matplotlib
import random, time
from Token import *

class Autoencoder(object):
    def __init__(self,n_features ,learning_rate=0.5,n_hidden=[1000,500,250,2],alpha=0.0, decay_rate = 1.0):
        with tf.device('/gpu:0'):   ## ASUS-Joseph-18080601
            self.n_features = n_features

            self.weights = None
            self.biases = None
            self.saver = None

            self.graph = tf.Graph() # initialize new grap
            self.build(n_features, learning_rate,n_hidden,alpha, decay_rate) # building graph
            ## ASUS-Joseph-18080601 >>> 
            self.sess = tf.Session(graph=self.graph) # create session by the graph 
            #self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(log_device_placement=True)) # create session by the graph 
            ## ASUS-Joseph-18080601 <<<
            if RESTORE_MODEL_ENABLE:
                self.saver.restore(self.sess, IMPORT_PATH)


    def build(self,n_features ,learning_rate,n_hidden,alpha, decay_rate):
        with self.graph.as_default():
            ### Input
            self.train_features = tf.placeholder(tf.float32, shape=(None,n_features))
            self.train_targets  = tf.placeholder(tf.float32, shape=(None,n_features))

            ### Optimalization
            # build neurel network structure and get their predictions and loss
            self.y_, self.original_loss, _ = self.structure(
                                               features=self.train_features,
                                               targets=self.train_targets,
                                               n_hidden=n_hidden)

            # regularization loss
            # weight elimination L2 regularizer
            self.regularizer = tf.reduce_sum([tf.reduce_sum(
                        tf.pow(w,2)/(1+tf.pow(w,2))) for w in self.weights.values()]) \
                    / tf.reduce_sum(
                     [tf.size(w,out_type=tf.float32) for w in self.weights.values()])

            # total loss
            self.loss = self.original_loss + alpha * self.regularizer

            ###########################
            #
            # define training operation
            #
            ############################

            ## Choose an optimizer
            # ASUS-Joseph-18081304 >>>
            if decay_rate != 1.0:
                exp_learning_rate = tf.train.exponential_decay(
                    learning_rate = learning_rate, global_step = 1, decay_steps = 100, decay_rate = decay_rate, staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=exp_learning_rate)        #<---- current used 
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)        #<---- current used 

            
            # ASUS-Joseph-18081304 <<<

            #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)     # nan
            #self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=0.1)
            #self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum = 0.1)   #nan
            self.train_op = self.optimizer.minimize(self.loss)

            ### Prediction
            self.new_features = tf.placeholder(tf.float32, shape=(None,n_features))
            self.new_targets  = tf.placeholder(tf.float32, shape=(None,n_features))
            self.new_y_, self.new_original_loss, self.new_encoder = self.structure(
                                                          features=self.new_features,
                                                          targets=self.new_targets,
                                                          n_hidden=n_hidden)  
            self.new_loss = self.new_original_loss + alpha * self.regularizer

            if RESTORE_MODEL_ENABLE:
                self.saver = tf.train.Saver()
            else:
                ### Initialization
                self.init_op = tf.global_variables_initializer()
                if AUTO_SAVE_MODEL_ENABLE | FINISH_SAVE_MODEL_ENABLE:
                    self.saver = tf.train.Saver()


    def structure(self,features,targets,n_hidden):
        ### Variable
        if (not self.weights) and (not self.biases):
            self.weights = {}
            self.biases = {}

            n_encoder = [self.n_features]+n_hidden
            for i,n in enumerate(n_encoder[:-1]):
                self.weights['encode{}'.format(i+1)] = \
                    tf.Variable(tf.truncated_normal(
                        shape=(n,n_encoder[i+1]),
                        #mean = 0.5,                       # ASUS-Joseph-18081301 
                        stddev=0.1),dtype=tf.float32
                        )
                self.biases['encode{}'.format(i+1)] = \
                    tf.Variable(tf.zeros( shape=(n_encoder[i+1]) ),dtype=tf.float32
                    )

            n_decoder = list(reversed(n_hidden))+[self.n_features]
            for i,n in enumerate(n_decoder[:-1]):
                self.weights['decode{}'.format(i+1)] = \
                    tf.Variable(tf.truncated_normal(
                        shape=(n,n_decoder[i+1]),
                        #mean = 0.5,                       # ASUS-Joseph-18081301
                        stddev=0.1),dtype=tf.float32
                        )
                self.biases['decode{}'.format(i+1)] = \
                    tf.Variable(tf.zeros( shape=(n_decoder[i+1]) ),dtype=tf.float32)                    

        ### Structure
        ### choose an activation function 

        # ASUS-Joseph-18081601 >>>
        #activation = tf.nn.selu
        activation = tf.nn.elu  #<--- current best
        #activation = tf.nn.relu    #<-- all black
        #activation = tf.nn.softsign
        #activation = tf.nn.sigmoid
        # ASUS-Joseph-18081601 <<<

        encoder = self.getDenseLayer(features,
                                     self.weights['encode1'],
                                     self.biases['encode1'],
                                     activation=activation)

        for i in range(1,len(n_hidden)-1):
            encoder = self.getDenseLayer(encoder,
                        self.weights['encode{}'.format(i+1)],
                        self.biases['encode{}'.format(i+1)],
                        #activation=tf.nn.dropout(keep_prob = 0.9)) 
                        activation=activation)   

        encoder = self.getDenseLayer(encoder,
                        self.weights['encode{}'.format(len(n_hidden))],
                        self.biases['encode{}'.format(len(n_hidden))]) 

        decoder = self.getDenseLayer(encoder,
                                     self.weights['decode1'],
                                     self.biases['decode1'],
                                     #activation=tf.nn.softsign)  # ASUS-Joseph-18081302 !!!
                                     activation=activation) 

        for i in range(1,len(n_hidden)-1):
            decoder = self.getDenseLayer(decoder,
                        self.weights['decode{}'.format(i+1)],
                        self.biases['decode{}'.format(i+1)],
                        #activation=tf.nn.dropout(keep_prob = 0.9))
                        #activation=tf.nn.softsign)  # ASUS-Joseph-18081302 !!!
                        activation=activation) 

        y_ =  self.getDenseLayer(decoder,
                        self.weights['decode{}'.format(len(n_hidden))],
                        self.biases['decode{}'.format(len(n_hidden))],
						# ASUS-Joseph-18081201 >>>
                        #activation=tf.nn.relu)
                        activation=activation) 	
						# ASUS-Joseph-18081201 <<<

        #loss = tf.reduce_mean(tf.pow(targets - y_, 2))
        #loss = tf.losses.absolute_difference(targets, y_) 
        loss = tf.contrib.losses.mean_squared_error(targets, y_)    #<--- current best: Edge is smoother than absolute_difference
        #loss = tf.losses.log_loss(targets, y_, epsilon=1e-01)

        return (y_,loss,encoder)

    def getDenseLayer(self,input_layer,weight,bias,activation=None):
        x = tf.add(tf.matmul(input_layer,weight),bias)
        if activation:
            x = activation(x)
        return x


    def fit(self,X,Y,epochs=10,validation_data=None,test_data=None,batch_size=None):
        X = self._check_array(X)
        Y = self._check_array(Y)

        N = X.shape[0]
        random.seed(9000)
        if not batch_size:
            batch_size=N

        if not RESTORE_MODEL_ENABLE:        # ASUS-Joseph-18082402
            self.sess.run(self.init_op)     

        for epoch in range(epochs):
            print("Epoch %2d/%2d: "%(epoch+1,epochs))
            start_time = time.time()

            # mini-batch gradient descent
            index = [i for i in range(N)]
            #random.shuffle(index)		# ASUS-Joseph-test  the batch size 288 is the X-axis coordination which cannot be shuffled
            while len(index)>0:
                index_size = len(index)
                batch_index = [index.pop() for _ in range(min(batch_size,index_size))]     

                feed_dict = {self.train_features: X[batch_index,:],
                             self.train_targets: Y[batch_index,:]}
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                print("[%d/%d] loss = %9.4f     " % ( N-len(index), N, loss ), end='\r')


            # evaluate at the end of this epoch
            msg_valid = ""
            if validation_data is not None:
                val_loss = self.evaluate(validation_data,validation_data)
                msg_valid = ", val_loss = %9.4f" % ( val_loss )
            
            train_loss = self.evaluate(X,Y)         
            print("[%d/%d] %ds loss = %9.4f %s" % ( N, N, time.time()-start_time,
                                                   train_loss, msg_valid ))

        if test_data is not None:
            test_loss = self.evaluate(test_data[0],test_data[1])
            print("test_loss = %9.4f" % (test_loss))

    def encode(self,X):
        X = self._check_array(X)
        return self.sess.run(self.new_encoder, feed_dict={self.new_features: X})

    def predict(self,X):
        X = self._check_array(X)
        return self.sess.run(self.new_y_, feed_dict={self.new_features: X})

    def evaluate(self,X,Y):
        # ASUS-Joseph-18081601 >>>
        """
        X = self._check_array(X)
        loss_sum = 0
        for i in range(len(X)):
            loss_sum = loss_sum + self.sess.run(self.new_loss, feed_dict={self.new_features: X[i],
                                                       self.new_targets: Y[i]})
        """
        loss_sum = 0
        loss_sum = loss_sum + self.sess.run(self.new_loss, feed_dict={self.new_features: X,
                                                       self.new_targets: Y})
        
        # ASUS-Joseph-18081601 <<<
        return loss_sum/len(X)

    def _check_array(self,ndarray):
        ndarray = np.array(ndarray)
        if len(ndarray.shape)==1: ndarray = np.reshape(ndarray,(1,ndarray.shape[0]))
        return ndarray

    #ASUS-Joseph-18082401 >>>
    def save_model(self, export_path):
        #saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, export_path)
        print("Save modoel variables at ", save_path)

    #ASUS-Joseph-18082401 <<<