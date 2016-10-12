from collections import defaultdict
import numpy as np
import tensorflow as tf

class basicRNN_Seq2Seq(object):

    def __init__(self, timesteps ,batch_size, forget_bias = 1.0 , input_bias = 1.0 , cell_bias = 1.0 , output_bias = 1.0, activation = tanh, input_data_total,output_data_total ):
        self.timesteps = timesteps
        self.forget_bias  = forget_bias
        self.input_bias = input_bias
        self.output_bias = output_bias
        self.cell_bias = cell_bias
        self._activation = activation
        self.input_data_total = input_data_total
        self.output_data_total = output_data_total
        self.batch_size  = batch_size

        # introduce inputs in the above function
        # inputs should be of form [timesteps, batchsize, feature_size]

        [ self.rows_raw_data, self.columns_raw_data]  = tf.shape(input_data_total)

        # add placeholders
        self.xts = tf.placeholder(tf.float32, shape =[batch_size,self.timesteps,feature_size])
        self.yts = tf.placeholder(tf.float32, shape = [batch_size, self.timesteps, 1])

        # add trainable variables
        with tf.variable_scope('rnn_cell'):

            self.W  = tf.Variable(tf.truncated_normal([batch_size,feature_size], sttdev = 1.0),name = 'W')
            self.U  = tf.Variable(tf.truncated_normal([1,1]),sttdev = 1.0, name = 'U')
            self.state_bias = tf.Variable(tf.ones(shape=[1,1]), name = 'state_bias')


    def forward_pass_one_timestep(self, inputs_cell, state_cell):

        # basically basicRNN cell


        # check if it's necessary since we have defiuned the variables as global
        #with tf.variable_scope('rnn_cell'):
            #self.W = tf.get_variable('')

        state = tf.tanh(tf.add(tf.add(tf.matmul(inputs_cell, self.W),tf.matmul(state_cell,self.U)), self.state_bias))
        return state

    def forward_all_timesteps(self, rnn_batch_input):

        rnn_inputs  =tf.unpack(rnn_batch_input, axis=0)

        ## TODO - define init_state somewhere as being just a matrix of zeros

        state = init_state
        rnn_outputs = []
        for rnn_input in rnn_inputs:
            new_state = forward_pass_one_timestep(rnn_input, state)
            rnn_outputs.append(new_state)

        return rnn_outputs

    def devise_loss_function(self, x_temp,y_temp):

        current_prediction = forward_all_timesteps(x_temp)
        ## the RMSE loss function

        mse_for_current_batch  = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(current_prediction, y_temp))))

    def generate_batch(self):
        data_x = np.zeros([self.timesteps,self.batch_size,self.columns_raw_data],dtype=np.int32)
        data_y = np.zeros([self.timesteps,self.batch_size,1],dtype=np.int32)

        lista_numerica_raw_data = np.arange(self.rows_raw_data)
        lista_numerica_raw_data_permutata = np.random.shuffle(lista_numerica_raw_data)
        lista_batch = lista_numerica_raw_data_permutata[:batch_size]

        raw_data_batch = self.input_data_total[lista_batch, :]
        for i in np.range(self.timesteps):
            control_temp = 0
            for j in lista_batch:
                data_x[i,control_temp,:] = self.input_data_total[j-(self.timesteps-i),:]
                control_temp+=1

                data_y[i,control_temp,1] = self.output_data_total[j-(self.timesteps-i),:]


        return data_x, data_y

    def train_network(self,num_epochs):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for __ in np.range(num_epochs):

                # crete the current batch
                rows_data_total
                lista_total_data = np.arrang

                # get current batch
                batch_x , batch_y = generate_batch()

                current_prediction = forward_all_timesteps(batch_x)
                ## the RMSE loss function
                lossssqrt(tf.reduce_mean(tf.square(tf.sub(current_prediction, y_temp))))











