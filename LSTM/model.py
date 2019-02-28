import numpy as np
import tensorflow as tf
import time


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
     # X in format(m,size) and Y in (m,num_of_class)
     m = X.shape[0]                  
     mini_batches = []
     np.random.seed(seed)
     permutation = list(np.random.permutation(m))
     shuffled_X = X[permutation]
     shuffled_Y = Y[permutation]
     num_complete_minibatches = int(np.floor(m/mini_batch_size)) 
     for k in range(0, num_complete_minibatches):
         mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
         mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
         mini_batch = (mini_batch_X, mini_batch_Y)
         mini_batches.append(mini_batch)
     
     if m % mini_batch_size != 0:
         mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m]
         mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m]
         mini_batch = (mini_batch_X, mini_batch_Y)
         mini_batches.append(mini_batch)
     
     return mini_batches

def makeOneHot(y):
     u=np.unique(y)
     x=np.zeros((len(u),len(y)))
     for i in range(len(y)):
         x[y[i],i]=1
     return x


tf.reset_default_graph()
class myLSTM:
    def __init__(self,max_len,num_class=2,num_hiddenStates=128):
        self.num_hiddenStates=num_hiddenStates
        self.max_len=max_len
        #self.input_dimension=input_dimension
        self.num_class=num_class;
    
    def LSTM_block(self,X,num_hiddenStates,Hprev=0,Cprev=0,device="/device:GPU:0",u=True):
    # hprev and cprev are of shape=(batch,num_hiddenStates)
    # x is of shape(batch,input dimension)
        #print("here")
        x=tf.transpose(X)
        hprev=tf.transpose(Hprev)
        cprev=tf.transpose(Cprev)
        inputx=tf.concat([x,hprev],axis=0)
        with tf.device(device):
            with tf.variable_scope("LSTMweights", reuse=u):
                Wi=tf.get_variable(shape=(num_hiddenStates,x.shape[0]+num_hiddenStates),name="Wi",initializer=tf.contrib.layers.xavier_initializer())
                Wf=tf.get_variable(shape=(num_hiddenStates,x.shape[0]+num_hiddenStates),name="Wf",initializer=tf.contrib.layers.xavier_initializer())
                Wc=tf.get_variable(shape=(num_hiddenStates,x.shape[0]+num_hiddenStates),name="Wc",initializer=tf.contrib.layers.xavier_initializer())
                Wo=tf.get_variable(shape=(num_hiddenStates,x.shape[0]+num_hiddenStates),name="Wo",initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope("LSTMbiases", reuse=u):
                bi=tf.get_variable(shape=(num_hiddenStates,1),name="bi",initializer=tf.zeros_initializer)
                bf=tf.get_variable(shape=(num_hiddenStates,1),name="bf",initializer=tf.zeros_initializer)
                bc=tf.get_variable(shape=(num_hiddenStates,1),name="bc",initializer=tf.zeros_initializer)
                bo=tf.get_variable(shape=(num_hiddenStates,1),name="bo",initializer=tf.zeros_initializer)
        
        ## calculate functions
            
        Ft=tf.sigmoid( tf.matmul(Wf,inputx)+bf )
        it=tf.sigmoid( tf.matmul(Wi,inputx)+bi )
        ot=tf.sigmoid( tf.matmul(Wo,inputx)+bo )
        Cp=tf.tanh( tf.matmul(Wc,inputx)+bc )
        Ct=Ft*cprev + it*Cp
        ht=tf.transpose(ot*tf.tanh(Ct))
        Ct=tf.transpose(Ct)
        return ht,Ct

    def build_model(self,vocabulary_size,embedding_size,learning_rate=0.001,optimizer="adam"):
        
        self.raw_input=tf.placeholder(shape=(None,self.max_len),dtype=tf.int32,name="RAWinput")
        #self.input=tf.placeholder(shape=(None,self.input_dimension,self.max_len),dtype=tf.float32,name="input")
        self.y=tf.placeholder(shape=(None,self.num_class),dtype=tf.float32,name="y")
        self.cprev=tf.placeholder(shape=(None,self.num_hiddenStates),dtype=tf.float32,name="cprev")
        self.hprev=tf.placeholder(shape=(None,self.num_hiddenStates),dtype=tf.float32,name="hprev")
        self.keep_prob = tf.placeholder(tf.float32)

        
        with tf.variable_scope("embeddings"):
            embeddings =tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, self.raw_input)
        embed=tf.transpose(embed, [0, 2, 1]) #here embed is of size (m, vector_size, num_steps)

        [hprev,cprev]=self.LSTM_block(embed[:,:,0],self.num_hiddenStates,self.hprev,self.cprev,u=False)
        hprev=tf.nn.dropout(hprev,self.keep_prob,name="dropOutH0")
        #cprev=tf.nn.dropout(cprev,self.keep_prob,name="dropOutC0")

        ##### I should give zero vector as the first cprev
        for i in range(1,self.max_len):
            [hprev,cprev]=self.LSTM_block(embed[:,:,i],self.num_hiddenStates,hprev,cprev)
            name="dropOutH"+str(i)
            hprev=tf.nn.dropout(hprev,self.keep_prob,name=name)
            #cprev=tf.nn.dropout(cprev,self.keep_prob,name=name)
            #print(hprev,cprev)

        hprev=tf.transpose(hprev) ##
        cprev=tf.transpose(cprev) ## here both hprev and cprev are of shape(num_hiddenState,batch)
        #print(hprev,cprev)
        #l=hprev.get_shape().as_list()
        with tf.variable_scope("denseLayer"):
            weights=tf.get_variable(name="finalWeights",shape=(self.num_class,self.num_hiddenStates),initializer=tf.random_normal_initializer)
        #with tf.device("/gpu:0"):
        biases = tf.get_variable(name='biases',shape=[self.num_class,1], initializer=tf.constant_initializer(0.1))
        self.final_output = tf.nn.relu(tf.matmul(weights,hprev)+biases)
        
        ytranspose=tf.transpose(self.y)
        self.inference=tf.nn.softmax(self.final_output,axis=0)
        predict_op = tf.argmax(self.inference, 0)
        correct_prediction = tf.equal(predict_op, tf.argmax(ytranspose, 0))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print("Computational graph construction finished---\n")
        self.loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=ytranspose,logits=self.final_output,dim=0)
        #self.loss=loss
        self.final_loss=tf.reduce_mean(self.loss)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.final_loss)
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("----------Model Ready-------\n")

        
    def train(self,X_train,Y_train,num_epochs=10,minibatch_size=32,print_cost=True,droupout=1):
        ## X_train should of shape (m, max_len) and Y_train should be of shape=(m,NumClasses)
        init_cprev=np.zeros((minibatch_size,self.num_hiddenStates))
        init_hprev=np.zeros((minibatch_size,self.num_hiddenStates))

        m=X_train.shape[0]
        seed=0
        for epoch in range(num_epochs):
            start_time_opoch = time.time()
            epoch_cost = 0.0
            num_minibatches = int(m / minibatch_size) 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                init_cprev=np.zeros((len(minibatch_X),self.num_hiddenStates))
                init_hprev=np.zeros((len(minibatch_X),self.num_hiddenStates))

                feed_dictt={self.raw_input:minibatch_X, self.y:minibatch_Y,self.cprev:init_cprev,self.hprev:init_hprev,self.keep_prob:droupout}
                _ , temp_cost = self.sess.run([self.optimizer, self.final_loss], feed_dict=feed_dictt)
                #test_writer.add_summary(summary, iterr)
                #temp_cost = sess.run(self.loss, feed_dict=feed_dictt)
                #print("here")
                epoch_cost += temp_cost / num_minibatches
                #costs.append(minibatch_cost)

            print("Time Of the Execution of epoch %i = %s seconds ---" % (epoch,time.time() - start_time_opoch))
            print ("Cost after epoch %i: %f" % (epoch, epoch_cost))


    def predict(self,x):
        m=x.shape[0]
        init_cprev=np.zeros((m,self.num_hiddenStates))
        init_hprev=np.zeros((m,self.num_hiddenStates))
        #sess.run(tf.global_variables_initializer())
        feed_dictt={self.raw_input:x,self.cprev:init_cprev,self.hprev:init_hprev,self.keep_prob:1}
        predictions = self.sess.run(self.final_output, feed_dict=feed_dictt)
        return predictions
    
    def evaluate(self,x,Y):
        m=x.shape[0]
        init_cprev=np.zeros((m,self.num_hiddenStates))
        init_hprev=np.zeros((m,self.num_hiddenStates))
        feed_dictt={self.raw_input:x,self.y:Y,self.cprev:init_cprev,self.hprev:init_hprev,self.keep_prob:1}
        accuracy = self.sess.run(self.accuracy, feed_dict=feed_dictt)
        return accuracy

        
