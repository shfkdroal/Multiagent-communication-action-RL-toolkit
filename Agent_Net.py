
import sys
import turtle
import os

import tensorflow as tf
import random
import numpy as np
import ast
#import Extract_ExpertTrajactories
import graphic
import math
from Game_Param import *

def conv2d(input, kernel_size, stride, num_filter, name='conv2d'):
    with tf.compat.v1.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        #print(input.get_shape())
        filter_shape = [kernel_size, kernel_size, input.get_shape()[3], num_filter]

        W = tf.compat.v1.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.compat.v1.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d(input, W, stride_shape, padding='SAME') + b

def conv2d_transpose(input, kernel_size, stride, num_filter, name='conv2d_transpose'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [kernel_size, kernel_size, num_filter, input.get_shape()[3]]
        output_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, num_filter])

        W = tf.compat.v1.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.compat.v1.get_variable('b', [1, 1, 1, num_filter], initializer=tf.constant_initializer(0.0))
        return tf.nn.conv2d_transpose(input, W, output_shape, stride_shape, padding='SAME') + b


def fc(input, num_output, name='fc'):
    with tf.compat.v1.variable_scope(name):
        num_input = input.get_shape()[1]
        W = tf.compat.v1.get_variable('w', [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.compat.v1.get_variable('b', [num_output], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input, W) + b

def batch_norm(input, is_training):
    # out = tf.contrib.layers.batch_norm(input, decay=0.99, center=True, scale=True,
    #                                    is_training=is_training, updates_collections=None)
    layer = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
    out = layer(input, is_training)
    return out

def leaky_relu(input, alpha=0.2):
    return tf.maximum(alpha * input, input)






class agent_net(object):
    def __init__(self, physical_observation_shape=None, should_load=False, length_seq_w=None):


        #self.device = "/cpu:0"
        self.device = "/gpu:0"
        self.device2 = "/gpu:1"
        self.device3 = "/gpu:2"

        #CONFIG PARAMS
        self.INTRA_OP_THREADS = 1
        self.INTER_OP_THREADS = 1
        self.SOFT_PLACEMENT = True
        self.SEED = None #48  # Set to None for random seed.
        tf.compat.v1.set_random_seed(self.SEED)
        self.config = tf.compat.v1.ConfigProto(allow_soft_placement=self.SOFT_PLACEMENT,
                intra_op_parallelism_threads=self.INTRA_OP_THREADS,
                inter_op_parallelism_threads=self.INTER_OP_THREADS)
        self.checkpoint = './model/agent_policy_model'
        self.should_load = should_load
        if self.should_load:
            self.checkpoint += str(model_itr)

        #dimension
        self.physical_input_dim = physical_observation_shape #S, A, hf real
        self.physical_output_dim = 20
        self.physical_input_w = self.physical_input_dim[0]*4
        self.physical_input_h = self.physical_input_dim[1]
        self.physical_input_dim_total = self.physical_input_w * self.physical_input_h + 1



        self.vocab_size = len(Vocab)
        self.emotion_size = len(Emotions)
        self.com_action_size = self.vocab_size + self.emotion_size

        self.latent_output_size_p = 100

        self.emotion_dim = 4

        self.n_word_input_dim = 3
        self.length_w = length_seq_w #length of the sequence of words
        # number of units in RNN cell
        self.n_hidden_w = 100
        # RNN output node weights and biases
        self.latent_output_size_w = Num_Agents * self.physical_input_dim_total #predicted S, A dimension

        self.weights_w = {
            'out': tf.Variable(tf.random.normal([self.n_hidden_w, self.latent_output_size_w]))
        }
        self.biases_w = {
            'out': tf.Variable(tf.random.normal([self.latent_output_size_w]))
        }



        ######################


        self.small_num = 1e-20
        # self.num_epoch = NE
        # self.batch_size = BatchSize #=32

        self.indices = [0]*BatchSize

        self.learning_rate = 8e-6

        self.P_called = False
        self.P_called2 = False
        self.P_called3 = False
        self.FR_called = False
        self.FR_called2 = False
        self.FR_called3 = False
        self.EPL_called = False
        self.EPL_called2 = False
        self.EPL_called3 = False
        self.mental_state_called = False
        self.P_W_latent_layer_called = False
        self.P_W_latent_layer_called2 = False
        self.P_W_latent_layer_called3 = False
        self.physical_action_called  = False
        self.physical_action_called2 = False
        self.physical_action_called3 = False
        self.com_action_called = False
        self.com_action_called2 = False
        self.com_action_called3 = False
        self.com_action_called_R = False
        self.com_action_called_R2 = False
        self.com_action_called_R3 = False
        self.dec_called = False

        #[None, self.code_size] -> 32, 213, 1, 1
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_training')
        self.learning_rate_shifting = tf.compat.v1.placeholder(tf.float32, name='lt')

        self._init_ops()


    # Define operations
    def _init_ops(self):
        with tf.device(self.device):
            with tf.name_scope('agent_feed_data'):
                # time length * physical input dim
                self.type_input = tf.compat.v1.placeholder(tf.float32,
                                                 [None, self.physical_input_dim[0], self.physical_input_dim[1]],
                                                 name='types')
                self.hp_input = tf.compat.v1.placeholder(tf.float32,
                                               [None, self.physical_input_dim[0], self.physical_input_dim[1]],
                                               name='hp')
                self.angle_input = tf.compat.v1.placeholder(tf.float32,
                                                  [None, self.physical_input_dim[0], self.physical_input_dim[1]],
                                                  name='angles')
                self.action_took_input = tf.compat.v1.placeholder(tf.float32,
                                                        [None, self.physical_input_dim[0], self.physical_input_dim[1]],
                                                        name='actions')

                self.hf_value = tf.compat.v1.placeholder(tf.float32, [None, 1], name='hf_value')

                self.physical_input = tf.concat([tf.concat([tf.concat([self.type_input, self.hp_input], axis=1),
                                                            self.angle_input], axis=1), self.action_took_input],
                                                                               axis=1 , name='physical_input')
                # time length * word dimension
                self.words_input1 = tf.compat.v1.placeholder(tf.float32, [None, self.length_w, self.n_word_input_dim],
                                                   name='word_input1')
                self.words_input2 = tf.compat.v1.placeholder(tf.float32, [None, self.length_w, self.n_word_input_dim],
                                                   name='word_input2')
                self.words_input3 = tf.compat.v1.placeholder(tf.float32, [None, self.length_w, self.n_word_input_dim],
                                                   name='word_input3')

                # emotion input dimension
                self.emotion_input1 = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input1')
                self.emotion_input2 = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input2')
                self.emotion_input3 = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input3')

                # self.emotion_input1_prevfake = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input1')
                # self.emotion_input2_prevfake = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input2')
                # self.emotion_input3_prevfake = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input3')
                #
                # self.emotion_input1_prevreal = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input1')
                # self.emotion_input2_prevreal = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input2')
                # self.emotion_input3_prevreal = tf.compat.v1.placeholder(tf.float32, [None, self.emotion_dim], name='emotion_input3')

                self.reward_got = tf.compat.v1.placeholder(tf.float32, [None, 1])
                self.com_reward = tf.compat.v1.placeholder(tf.float32, [None, 1])

            with tf.name_scope('agent_policy'):
                # RNN words net shared among agents
                self.words_results1 = self.RNN(self.words_input1, self.weights_w, self.biases_w)
                self.words_results2 = self.RNN(self.words_input2, self.weights_w, self.biases_w)
                self.words_results3 = self.RNN(self.words_input3, self.weights_w, self.biases_w)

                # self.physical_result = self.RNNPhysical(self.physical_input, self.weights_p, self.biases_p)
                self.physical_result = self.Physical_Interpret(self.physical_input, self.hf_value)
                self.physical_result2 = self.Physical_Interpret2(self.physical_input, self.hf_value)
                self.physical_result3 = self.Physical_Interpret3(self.physical_input, self.hf_value)

                self.p_w_latent_layer, dim1 = \
                    self.P_W_latent_layer(self.physical_result, self.words_results1, self.words_results2,
                                          self.words_results3)
                self.p_w_latent_layer2, dim1 = \
                    self.P_W_latent_layer2(self.physical_result2, self.words_results1, self.words_results2,
                                          self.words_results3)
                self.p_w_latent_layer3, dim1 = \
                    self.P_W_latent_layer3(self.physical_result3, self.words_results1, self.words_results2,
                                          self.words_results3)

                self.real_emotion_1 = self.FakeEmotion_to_RealEmotion(self.physical_result, self.emotion_input1)
                self.real_emotion_2 = self.FakeEmotion_to_RealEmotion(self.physical_result, self.emotion_input2)
                self.real_emotion_3 = self.FakeEmotion_to_RealEmotion(self.physical_result, self.emotion_input3)

                self.real_emotion_21 = self.FakeEmotion_to_RealEmotion2(self.physical_result2, self.emotion_input1)
                self.real_emotion_22 = self.FakeEmotion_to_RealEmotion2(self.physical_result2, self.emotion_input2)
                self.real_emotion_23 = self.FakeEmotion_to_RealEmotion2(self.physical_result2, self.emotion_input3)

                self.real_emotion_31 = self.FakeEmotion_to_RealEmotion3(self.physical_result3, self.emotion_input1)
                self.real_emotion_32 = self.FakeEmotion_to_RealEmotion3(self.physical_result3, self.emotion_input2)
                self.real_emotion_33 = self.FakeEmotion_to_RealEmotion3(self.physical_result3, self.emotion_input3)

                self.e_p_latent_layer, dim2 = self.E_P_latent_layer(self.physical_result, self.real_emotion_1,
                                                                    self.real_emotion_2, self.real_emotion_3,
                                                                    self.emotion_input1, self.emotion_input2,
                                                                    self.emotion_input3)
                self.e_p_latent_layer2, dim2 = self.E_P_latent_layer2(self.physical_result2, self.real_emotion_21,
                                                                    self.real_emotion_22, self.real_emotion_23,
                                                                    self.emotion_input1, self.emotion_input2,
                                                                    self.emotion_input3)
                self.e_p_latent_layer3, dim2 = self.E_P_latent_layer3(self.physical_result3, self.real_emotion_31,
                                                                    self.real_emotion_32, self.real_emotion_33,
                                                                    self.emotion_input1, self.emotion_input2,
                                                                    self.emotion_input3)

                self.internal_mental_state = self.emotions_to_mental_state(self.e_p_latent_layer)
                self.internal_mental_state2 = self.emotions_to_mental_state(self.e_p_latent_layer2)
                self.internal_mental_state3 = self.emotions_to_mental_state(self.e_p_latent_layer3)


                self._action = self.physical_action(self.p_w_latent_layer, self.e_p_latent_layer,
                                                    self.internal_mental_state, dim1, dim2)
                self._action2 = self.physical_action2(self.p_w_latent_layer2, self.e_p_latent_layer2,
                                                    self.internal_mental_state2, dim1, dim2)
                self._action3 = self.physical_action3(self.p_w_latent_layer3, self.e_p_latent_layer3,
                                                    self.internal_mental_state3, dim1, dim2)

                # self._com_action = self.com_action_real(self._com_action_r, self.e_p_latent_layer,
                #                                         self.p_w_latent_layer, dim1, dim2)
                # self._com_action2 = self.com_action2_real(self._com_action2_r, self.e_p_latent_layer2,
                #                                           self.p_w_latent_layer2, dim1, dim2)
                # self._com_action3 = self.com_action3_real(self._com_action3_r, self.e_p_latent_layer3,
                #                                           self.p_w_latent_layer3, dim1, dim2)

                self._com_action = self.com_action(self.p_w_latent_layer, self.e_p_latent_layer,
                                                   self.internal_mental_state3, dim1, dim2)
                self._com_action2 = self.com_action2(self.p_w_latent_layer2, self.e_p_latent_layer2,
                                                   self.internal_mental_state3, dim1, dim2)
                self._com_action3 = self.com_action3(self.p_w_latent_layer3, self.e_p_latent_layer3,
                                                   self.internal_mental_state3, dim1, dim2)


                # self.real_emotion_1_comp_ = \
                #     self.FakeEmotion_to_RealEmotion(self.physical_result, self.emotion_input1_prevfake) #self.emotion_input1_performed_real
                #
                # self.real_emotion_22_comp_ = \
                #     self.FakeEmotion_to_RealEmotion2(self.physical_result2, self.emotion_input2_prevfake)
                #
                # self.real_emotion_33_comp_ = \
                #     self.FakeEmotion_to_RealEmotion3(self.physical_result3, self.emotion_input3_prevfake)



        with tf.device(self.device):
            policy_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='agent_policy')
            policy_saver = tf.compat.v1.train.Saver(policy_variables, max_to_keep=100)

        with tf.device(self.device):
            # self.dis_loss_op = None
            self._com_action_comp = tf.reduce_max(self._com_action, axis=1)
            self._com_action_comp = tf.reduce_max(self._com_action_comp, axis=1)
            self._action_comp = tf.reduce_max(self._action, axis=1)
            #self._action_comp = tf.reduce_max(self._action_comp, axis=1)
            self._action_comp = tf.reshape(self._action_comp, [-1, 1])
            self._com_action_comp = tf.reshape(self._com_action_comp, [-1, 1])
            self.loss_op = self._loss2(self.reward_got, self._action_comp) + \
                           self._loss2(self.com_reward, self._com_action_comp) - self.Entropy(self._action)\
                           - self.Entropy(self._com_action)
                           # 0.1*self._loss2(self.real_emotion_1_comp_, self.emotion_input1_prevreal)

            optimizer1 = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate_shifting) #self.learning_rate) #20
            # self.dis_train_op = dis_optimizer.minimize(self.dis_loss_op)
            # print(dis_train_vars)

            # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            train_vars_PI = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "PI")
            train_vars_DEC = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "DEC")
            train_vars_PWL = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "PWL")
            train_vars_FR = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "FR")
            train_vars_EPL = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "EPL")
            train_vars_EM = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "EM")
            train_vars_PA = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "PA")
            train_vars_CA = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "CA")
            # train_vars_CAR = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            #                                   "CAR")
            # print(gen_train_vars)
            # self.gen_train_op = gen_optimizer.minimize(self.gen_loss_op)
            variables = train_vars_PI + train_vars_DEC + train_vars_PWL + train_vars_FR + train_vars_EPL + \
                        train_vars_EM + train_vars_PA + train_vars_CA #+ train_vars_CAR
            self.train_op = optimizer1.minimize(self.loss_op, var_list=variables)

        with tf.device(self.device2):
            self._com_action_comp2 = tf.reduce_max(self._com_action2, axis=1)
            self._com_action_comp2 = tf.reduce_max(self._com_action_comp2, axis=1)
            self._action_comp2 = tf.reduce_max(self._action2, axis=1)
            #self._action_comp = tf.reduce_max(self._action_comp, axis=1)
            self._action_comp2 = tf.reshape(self._action_comp2, [-1, 1])
            self._com_action_comp2 = tf.reshape(self._com_action_comp2, [-1, 1])
            self.loss_op2 = self._loss2(self.reward_got, self._action_comp2) + \
                           self._loss2(self.com_reward, self._com_action_comp2) - self.Entropy(self._action2)\
                           - self.Entropy(self._com_action2)
                           # 0.1*self._loss2(self.real_emotion_22_comp_, self.emotion_input2_prevreal)
            optimizer1 = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate_shifting) #self.learning_rate) #20
            # self.dis_train_op = dis_optimizer.minimize(self.dis_loss_op)
            # print(dis_train_vars)

            # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            train_vars_PI = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "PI2")
            train_vars_DEC = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "DEC")
            train_vars_PWL = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "PWL2")
            train_vars_FR = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "FR2")
            train_vars_EPL = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "EPL2")
            train_vars_EM = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "EM")
            train_vars_PA = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "PA2")
            train_vars_CA = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "CA2")
            # train_vars_CAR2 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            #                                   "CAR2")
            # print(gen_train_vars)
            # self.gen_train_op = gen_optimizer.minimize(self.gen_loss_op)
            variables = train_vars_PI + train_vars_DEC + train_vars_PWL + train_vars_FR + train_vars_EPL + \
                        train_vars_EM + train_vars_PA + train_vars_CA #+ train_vars_CAR2
            self.train_op2 = optimizer1.minimize(self.loss_op2, var_list=variables)

        with tf.device(self.device3):
            self._com_action_comp3 = tf.reduce_max(self._com_action3, axis=1)
            self._com_action_comp3 = tf.reduce_max(self._com_action_comp3, axis=1)
            self._action_comp3 = tf.reduce_max(self._action3, axis=1)
            #self._action_comp = tf.reduce_max(self._action_comp, axis=1)
            self._action_comp3 = tf.reshape(self._action_comp3, [-1, 1])
            self._com_action_comp3 = tf.reshape(self._com_action_comp3, [-1, 1])
            self.loss_op3 = self._loss2(self.reward_got, self._action_comp3) + \
                           self._loss2(self.com_reward, self._com_action_comp3) - self.Entropy(self._action3)\
                           - self.Entropy(self._com_action3)
                           # 0.1*self._loss2(self.real_emotion_33_comp_, self.emotion_input3_prevreal)
            optimizer1 = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate_shifting) #self.learning_rate) #20
            # self.dis_train_op = dis_optimizer.minimize(self.dis_loss_op)
            # print(dis_train_vars)

            # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            # gen_optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            train_vars_PI = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "PI3")
            train_vars_DEC = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "DEC")
            train_vars_PWL = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "PWL3")
            train_vars_FR = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "FR3")
            train_vars_EPL = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                               "EPL3")
            train_vars_EM = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "EM")
            train_vars_PA = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "PA3")
            train_vars_CA = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              "CA3")
            # train_vars_CAR3 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
            #                                   "CAR3")
            # print(gen_train_vars)
            # self.gen_train_op = gen_optimizer.minimize(self.gen_loss_op)
            variables = train_vars_PI + train_vars_DEC + train_vars_PWL + train_vars_FR + train_vars_EPL + \
                        train_vars_EM + train_vars_PA + train_vars_CA #+ train_vars_CAR3
            self.train_op3 = optimizer1.minimize(self.loss_op3, var_list=variables)

        with tf.name_scope('init'):
            init = tf.compat.v1.global_variables_initializer()
            self.saver = tf.compat.v1.train.Saver()

        with tf.device(self.device):
            self.sess = tf.compat.v1.Session(config=self.config)
            # add init, Jiali
            self.sess.run(init)
            #print(colored('init pro model with: {}'.format(self.checkpoint), 'magenta'))
            if self.should_load:
                policy_saver.restore(self.sess, self.checkpoint)






    def emotions_to_mental_state(self, E_P_latent):
        with tf.compat.v1.variable_scope('EM', reuse=self.mental_state_called):
            self.mental_state_called = True
            epl_fc1 = fc(tf.reshape(E_P_latent, [-1, self.latent_output_size_p]), 1, 'fc1')
            internal = tf.nn.sigmoid(epl_fc1)

            return internal

    def Physical_Interpret(self, physical_input, hf_value):
        with tf.compat.v1.variable_scope('PI', reuse=self.P_called):
            self.P_called =True
            # feedInput = tf.reshape(physical_input, [-1, self.physical_input_dim])
            input_shape = physical_input.get_shape()
            physical_input = tf.reshape(physical_input, [-1, input_shape[1], input_shape[2], 1])
            p_conv2 = conv2d(physical_input, 4, 1, 32, 'conv2')

            p_conv2 = tf.nn.max_pool(p_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p_batchnorm2 = batch_norm(p_conv2, self.is_train)
            p_lrelu2 = leaky_relu(p_batchnorm2)

            p_conv3 = conv2d(p_lrelu2, 4, 1, 16, 'conv3') #32 -> 64
            p_conv3 = tf.nn.max_pool(p_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p_batchnorm3 = batch_norm(p_conv3, self.is_train)
            p_lrelu3 = leaky_relu(p_batchnorm3)
            p_reshape = tf.reshape(p_lrelu3, [-1, 832]) #[-1, 128]
            p_reshape = tf.concat([p_reshape, hf_value], axis=1)
            p_fc = fc(p_reshape, self.physical_output_dim, 'p_fc')
            p_fc = tf.reshape(p_fc, [-1, self.physical_output_dim])
            return p_fc

    def Physical_Interpret2(self, physical_input, hf_value):
        with tf.compat.v1.variable_scope('PI2', reuse=self.P_called2):
            self.P_called2 =True
            # feedInput = tf.reshape(physical_input, [-1, self.physical_input_dim])
            input_shape = physical_input.get_shape()
            physical_input = tf.reshape(physical_input, [-1, input_shape[1], input_shape[2], 1])
            p_conv2 = conv2d(physical_input, 4, 1, 32, 'conv2')

            p_conv2 = tf.nn.max_pool(p_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p_batchnorm2 = batch_norm(p_conv2, self.is_train)
            p_lrelu2 = leaky_relu(p_batchnorm2)

            p_conv3 = conv2d(p_lrelu2, 4, 1, 16, 'conv3') #32 -> 64
            p_conv3 = tf.nn.max_pool(p_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p_batchnorm3 = batch_norm(p_conv3, self.is_train)
            p_lrelu3 = leaky_relu(p_batchnorm3)
            p_reshape = tf.reshape(p_lrelu3, [-1, 832]) #[-1, 128]
            p_reshape = tf.concat([p_reshape, hf_value], axis=1)
            p_fc = fc(p_reshape, self.physical_output_dim, 'p_fc')
            p_fc = tf.reshape(p_fc, [-1, self.physical_output_dim])
            return p_fc

    def Physical_Interpret3(self, physical_input, hf_value):
        with tf.compat.v1.variable_scope('PI3', reuse=self.P_called3):
            self.P_called3 =True
            # feedInput = tf.reshape(physical_input, [-1, self.physical_input_dim])
            input_shape = physical_input.get_shape()
            physical_input = tf.reshape(physical_input, [-1, input_shape[1], input_shape[2], 1])
            p_conv2 = conv2d(physical_input, 4, 1, 32, 'conv2')

            p_conv2 = tf.nn.max_pool(p_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p_batchnorm2 = batch_norm(p_conv2, self.is_train)
            p_lrelu2 = leaky_relu(p_batchnorm2)

            p_conv3 = conv2d(p_lrelu2, 4, 1, 16, 'conv3') #32 -> 64
            p_conv3 = tf.nn.max_pool(p_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            p_batchnorm3 = batch_norm(p_conv3, self.is_train)
            p_lrelu3 = leaky_relu(p_batchnorm3)
            p_reshape = tf.reshape(p_lrelu3, [-1, 832]) #[-1, 128]
            p_reshape = tf.concat([p_reshape, hf_value], axis=1)
            p_fc = fc(p_reshape, self.physical_output_dim, 'p_fc')
            p_fc = tf.reshape(p_fc, [-1, self.physical_output_dim])
            return p_fc


    def FakeEmotion_to_RealEmotion(self, physical_result, fake_emotion):
        with tf.compat.v1.variable_scope('FR', reuse=self.FR_called):
            self.FR_called = True
            # shape_dim = fake_emotion.get_shape()
            # fake_emotion = tf.reshape(fake_emotion, [-1, shape_dim[1] * shape_dim[2]])
            feedInput = tf.concat([physical_result, fake_emotion], axis=1)
            fr_fc1 = fc(tf.reshape(feedInput, [-1, self.physical_output_dim + self.emotion_dim]), self.emotion_dim, 'fc1')
            fr_batchnorm1 = batch_norm(fr_fc1, self.is_train)
            fr_lrelu1 = leaky_relu(fr_batchnorm1)

            fr_fc2 = fc(fr_lrelu1, self.emotion_dim, 'fc1_1')
            fr_fc2_reshaped = tf.reshape(fr_fc2, [-1, self.emotion_dim])
            return fr_fc2_reshaped

    def FakeEmotion_to_RealEmotion2(self, physical_result, fake_emotion):
        with tf.compat.v1.variable_scope('FR2', reuse=self.FR_called2):
            self.FR_called2 = True
            # shape_dim = fake_emotion.get_shape()
            # fake_emotion = tf.reshape(fake_emotion, [-1, shape_dim[1] * shape_dim[2]])
            feedInput = tf.concat([physical_result, fake_emotion], axis=1)
            fr_fc1 = fc(tf.reshape(feedInput, [-1, self.physical_output_dim + self.emotion_dim]), self.emotion_dim, 'fc1')
            fr_batchnorm1 = batch_norm(fr_fc1, self.is_train)
            fr_lrelu1 = leaky_relu(fr_batchnorm1)

            fr_fc2 = fc(fr_lrelu1, self.emotion_dim, 'fc1_1')
            fr_fc2_reshaped = tf.reshape(fr_fc2, [-1, self.emotion_dim])
            return fr_fc2_reshaped

    def FakeEmotion_to_RealEmotion3(self, physical_result, fake_emotion):
        with tf.compat.v1.variable_scope('FR3', reuse=self.FR_called3):
            self.FR_called3 = True
            # shape_dim = fake_emotion.get_shape()
            # fake_emotion = tf.reshape(fake_emotion, [-1, shape_dim[1] * shape_dim[2]])
            feedInput = tf.concat([physical_result, fake_emotion], axis=1)
            fr_fc1 = fc(tf.reshape(feedInput, [-1, self.physical_output_dim + self.emotion_dim]), self.emotion_dim, 'fc1')
            fr_batchnorm1 = batch_norm(fr_fc1, self.is_train)
            fr_lrelu1 = leaky_relu(fr_batchnorm1)

            fr_fc2 = fc(fr_lrelu1, self.emotion_dim, 'fc1_1')
            fr_fc2_reshaped = tf.reshape(fr_fc2, [-1, self.emotion_dim])
            return fr_fc2_reshaped


    def P_W_latent_layer(self, physical_result, words_results1, words_results2, words_results3):
        with tf.compat.v1.variable_scope('PWL', reuse=self.P_W_latent_layer_called):
            self.P_W_latent_layer_called = True
            physical_result = tf.concat([physical_result, words_results1], axis=1)
            physical_result = tf.concat([physical_result, words_results2], axis=1)
            physical_result = tf.concat([physical_result, words_results3], axis=1)

            dim = 3*self.latent_output_size_w + self.physical_output_dim #self.physical_input_dim_total +
            feedInput = tf.reshape(physical_result, [-1, dim])
            dim = int(2*dim/3)
            fr_fc1 = fc(feedInput, dim, 'fc1')

            fr_batchnorm1 = batch_norm(fr_fc1, self.is_train)
            fr_lrelu1 = leaky_relu(fr_batchnorm1)

            fr_fc2 = fc(fr_lrelu1, dim, 'fc1_1')
            fr_fc2_reshaped = tf.reshape(fr_fc2, [-1, dim])
            return fr_fc2_reshaped, dim

    def P_W_latent_layer2(self, physical_result, words_results1, words_results2, words_results3):
        with tf.compat.v1.variable_scope('PWL2', reuse=self.P_W_latent_layer_called2):
            self.P_W_latent_layer_called2 = True
            physical_result = tf.concat([physical_result, words_results1], axis=1)
            physical_result = tf.concat([physical_result, words_results2], axis=1)
            physical_result = tf.concat([physical_result, words_results3], axis=1)

            dim = 3*self.latent_output_size_w + self.physical_output_dim #self.physical_input_dim_total +
            feedInput = tf.reshape(physical_result, [-1, dim])
            dim = int(2*dim/3)
            fr_fc1 = fc(feedInput, dim, 'fc1')

            fr_batchnorm1 = batch_norm(fr_fc1, self.is_train)
            fr_lrelu1 = leaky_relu(fr_batchnorm1)

            fr_fc2 = fc(fr_lrelu1, dim, 'fc1_1')
            fr_fc2_reshaped = tf.reshape(fr_fc2, [-1, dim])
            return fr_fc2_reshaped, dim

    def P_W_latent_layer3(self, physical_result, words_results1, words_results2, words_results3):
        with tf.compat.v1.variable_scope('PWL3', reuse=self.P_W_latent_layer_called3):
            self.P_W_latent_layer_called3 = True
            physical_result = tf.concat([physical_result, words_results1], axis=1)
            physical_result = tf.concat([physical_result, words_results2], axis=1)
            physical_result = tf.concat([physical_result, words_results3], axis=1)

            dim = 3*self.latent_output_size_w + self.physical_output_dim #self.physical_input_dim_total +
            feedInput = tf.reshape(physical_result, [-1, dim])
            dim = int(2*dim/3)
            fr_fc1 = fc(feedInput, dim, 'fc1')

            fr_batchnorm1 = batch_norm(fr_fc1, self.is_train)
            fr_lrelu1 = leaky_relu(fr_batchnorm1)

            fr_fc2 = fc(fr_lrelu1, dim, 'fc1_1')
            fr_fc2_reshaped = tf.reshape(fr_fc2, [-1, dim])
            return fr_fc2_reshaped, dim

    def E_P_latent_layer(self, physical_result, real_emotion1, real_emotion2, real_emotion3,
                         fake_emotion1, fake_emotion2, fake_emotion3):
        with tf.compat.v1.variable_scope('EPL', reuse=self.EPL_called):
            self.EPL_called = True

            fake_emotion = tf.concat([fake_emotion1, fake_emotion2], axis=1)
            fake_emotion = tf.concat([fake_emotion, fake_emotion3], axis=1)
            real_emotion = tf.concat([real_emotion1, real_emotion2], axis=1)
            real_emotion = tf.concat([real_emotion, real_emotion3], axis=1)

            feedInput = tf.concat([physical_result, fake_emotion], axis=1)
            feedInput = tf.concat([feedInput, real_emotion], axis=1)

            epl_fc1 = fc(tf.reshape(feedInput, [-1, (self.physical_output_dim + 6*self.emotion_dim)]),
                        4 * 4 * 64, 'fc1')

            epl_batchnorm1 = batch_norm(epl_fc1, self.is_train)
            epl_lrelu1 = leaky_relu(epl_batchnorm1)

            epl_fc2 = fc(epl_lrelu1, self.latent_output_size_p, 'fc2')

            epl_batchnorm2 = batch_norm(epl_fc2, self.is_train)
            epl_lrelu2 = leaky_relu(epl_batchnorm2)

            epl_fc2_reshaped = tf.reshape(epl_lrelu2, [-1, self.latent_output_size_p])

            return epl_fc2_reshaped, self.latent_output_size_p

    def E_P_latent_layer2(self, physical_result, real_emotion1, real_emotion2, real_emotion3,
                         fake_emotion1, fake_emotion2, fake_emotion3):
        with tf.compat.v1.variable_scope('EPL2', reuse=self.EPL_called2):
            self.EPL_called2 = True

            fake_emotion = tf.concat([fake_emotion1, fake_emotion2], axis=1)
            fake_emotion = tf.concat([fake_emotion, fake_emotion3], axis=1)
            real_emotion = tf.concat([real_emotion1, real_emotion2], axis=1)
            real_emotion = tf.concat([real_emotion, real_emotion3], axis=1)

            feedInput = tf.concat([physical_result, fake_emotion], axis=1)
            feedInput = tf.concat([feedInput, real_emotion], axis=1)

            epl_fc1 = fc(tf.reshape(feedInput, [-1, (self.physical_output_dim + 6*self.emotion_dim)]),
                        4 * 4 * 64, 'fc1')

            epl_batchnorm1 = batch_norm(epl_fc1, self.is_train)
            epl_lrelu1 = leaky_relu(epl_batchnorm1)

            epl_fc2 = fc(epl_lrelu1, self.latent_output_size_p, 'fc2')

            epl_batchnorm2 = batch_norm(epl_fc2, self.is_train)
            epl_lrelu2 = leaky_relu(epl_batchnorm2)

            epl_fc2_reshaped = tf.reshape(epl_lrelu2, [-1, self.latent_output_size_p])

            return epl_fc2_reshaped, self.latent_output_size_p

    def E_P_latent_layer3(self, physical_result, real_emotion1, real_emotion2, real_emotion3,
                         fake_emotion1, fake_emotion2, fake_emotion3):
        with tf.compat.v1.variable_scope('EPL3', reuse=self.EPL_called3):
            self.EPL_called3 = True

            fake_emotion = tf.concat([fake_emotion1, fake_emotion2], axis=1)
            fake_emotion = tf.concat([fake_emotion, fake_emotion3], axis=1)
            real_emotion = tf.concat([real_emotion1, real_emotion2], axis=1)
            real_emotion = tf.concat([real_emotion, real_emotion3], axis=1)

            feedInput = tf.concat([physical_result, fake_emotion], axis=1)
            feedInput = tf.concat([feedInput, real_emotion], axis=1)

            epl_fc1 = fc(tf.reshape(feedInput, [-1, (self.physical_output_dim + 6*self.emotion_dim)]),
                        4 * 4 * 64, 'fc1')

            epl_batchnorm1 = batch_norm(epl_fc1, self.is_train)
            epl_lrelu1 = leaky_relu(epl_batchnorm1)

            epl_fc2 = fc(epl_lrelu1, self.latent_output_size_p, 'fc2')

            epl_batchnorm2 = batch_norm(epl_fc2, self.is_train)
            epl_lrelu2 = leaky_relu(epl_batchnorm2)

            epl_fc2_reshaped = tf.reshape(epl_lrelu2, [-1, self.latent_output_size_p])

            return epl_fc2_reshaped, self.latent_output_size_p

    def physical_action(self, p_w_latent_layer, e_p_latent_layer, internal_state, dim1, dim2):
        with tf.compat.v1.variable_scope('PA', reuse=self.physical_action_called):
            self.physical_action_called = True
            feedInput = tf.concat([p_w_latent_layer, e_p_latent_layer], axis=1)
            feedInput = tf.concat([feedInput, internal_state], axis=1)

            pa_fc1 = fc(feedInput, int((dim1 + dim2)/2), 'fc1')
            #pa_batchnorm1 = batch_norm(pa_fc1, self.is_train)
            pa_lrelu1 = leaky_relu(pa_fc1)

            dim3 = 8
            #pa_fc2 = fc(pa_lrelu1, (dim3*Num_Agents), 'fc2')
            pa_fc2 = fc(pa_lrelu1, dim3, 'fc2')

            #pa_fc2_reshaped = tf.reshape(pa_fc2, [-1, dim3, Num_Agents])
            pa_fc2_reshaped = tf.reshape(pa_fc2, [-1, dim3])
            return tf.nn.sigmoid(pa_fc2_reshaped)

    def physical_action2(self, p_w_latent_layer, e_p_latent_layer, internal_state, dim1, dim2):
        with tf.compat.v1.variable_scope('PA2', reuse=self.physical_action_called2):
            self.physical_action_called2 = True
            feedInput = tf.concat([p_w_latent_layer, e_p_latent_layer], axis=1)
            feedInput = tf.concat([feedInput, internal_state], axis=1)

            pa_fc1 = fc(feedInput, int((dim1 + dim2)/2), 'fc1')
            #pa_batchnorm1 = batch_norm(pa_fc1, self.is_train)
            pa_lrelu1 = leaky_relu(pa_fc1)

            dim3 = 8
            #pa_fc2 = fc(pa_lrelu1, (dim3*Num_Agents), 'fc2')
            pa_fc2 = fc(pa_lrelu1, dim3, 'fc2')

            #pa_fc2_reshaped = tf.reshape(pa_fc2, [-1, dim3, Num_Agents])
            pa_fc2_reshaped = tf.reshape(pa_fc2, [-1, dim3])
            return tf.nn.sigmoid(pa_fc2_reshaped)

    def physical_action3(self, p_w_latent_layer, e_p_latent_layer, internal_state, dim1, dim2):
        with tf.compat.v1.variable_scope('PA3', reuse=self.physical_action_called3):
            self.physical_action_called3 = True
            feedInput = tf.concat([p_w_latent_layer, e_p_latent_layer], axis=1)
            feedInput = tf.concat([feedInput, internal_state], axis=1)

            pa_fc1 = fc(feedInput, int((dim1 + dim2)/2), 'fc1')
            #pa_batchnorm1 = batch_norm(pa_fc1, self.is_train)
            pa_lrelu1 = leaky_relu(pa_fc1)

            dim3 = 8
            #pa_fc2 = fc(pa_lrelu1, (dim3*Num_Agents), 'fc2')
            pa_fc2 = fc(pa_lrelu1, dim3, 'fc2')

            #pa_fc2_reshaped = tf.reshape(pa_fc2, [-1, dim3, Num_Agents])
            pa_fc2_reshaped = tf.reshape(pa_fc2, [-1, dim3])
            return tf.nn.sigmoid(pa_fc2_reshaped)

    def com_action(self, p_w_latent_layer, e_p_latent_layer, internal_state, dim1, dim2):
        with tf.compat.v1.variable_scope('CA', reuse=self.com_action_called):
            self.com_action_called = True
            feedInput = tf.concat([p_w_latent_layer, e_p_latent_layer], axis=1)
            feedInput = tf.concat([feedInput, internal_state], axis=1)
            ca_fc1 = fc(feedInput, int((dim1 + dim2)/2), 'fc1')

            ca_batchnorm1 = batch_norm(ca_fc1, self.is_train)
            ca_lrelu1 = leaky_relu(ca_batchnorm1)

            dim3 = (len(Vocab) + 3 * len(Emotions))
            ca_fc2 = fc(ca_lrelu1, (dim3*Num_Agents), 'fc2')

            ca_fc2_reshaped = tf.reshape(ca_fc2, [-1, dim3, Num_Agents])
            return tf.nn.sigmoid(ca_fc2_reshaped)

    def com_action2(self, p_w_latent_layer, e_p_latent_layer, internal_state, dim1, dim2):
        with tf.compat.v1.variable_scope('CA2', reuse=self.com_action_called2):
            self.com_action_called2 = True
            feedInput = tf.concat([p_w_latent_layer, e_p_latent_layer], axis=1)
            feedInput = tf.concat([feedInput, internal_state], axis=1)
            ca_fc1 = fc(feedInput, int((dim1 + dim2)/2), 'fc1')

            ca_batchnorm1 = batch_norm(ca_fc1, self.is_train)
            ca_lrelu1 = leaky_relu(ca_batchnorm1)

            dim3 = (len(Vocab) + 3 * len(Emotions))
            ca_fc2 = fc(ca_lrelu1, (dim3*Num_Agents), 'fc2')

            ca_fc2_reshaped = tf.reshape(ca_fc2, [-1, dim3, Num_Agents])
            return tf.nn.sigmoid(ca_fc2_reshaped)

    def com_action3(self, p_w_latent_layer, e_p_latent_layer, internal_state, dim1, dim2):
        with tf.compat.v1.variable_scope('CA3', reuse=self.com_action_called3):
            self.com_action_called3 = True
            feedInput = tf.concat([p_w_latent_layer, e_p_latent_layer], axis=1)
            feedInput = tf.concat([feedInput, internal_state], axis=1)
            ca_fc1 = fc(feedInput, int((dim1 + dim2)/2), 'fc1')

            ca_batchnorm1 = batch_norm(ca_fc1, self.is_train)
            ca_lrelu1 = leaky_relu(ca_batchnorm1)

            dim3 = (len(Vocab) + 3 * len(Emotions))
            ca_fc2 = fc(ca_lrelu1, (dim3*Num_Agents), 'fc2')

            ca_fc2_reshaped = tf.reshape(ca_fc2, [-1, dim3, Num_Agents])
            return tf.nn.sigmoid(ca_fc2_reshaped)

    # def com_action_real(self, _com_action_, e_p_latent_layer, p_w_latent_layer, dim1, dim2):
    #     with tf.compat.v1.variable_scope('CAR', reuse=self.com_action_called_R):
    #         self.com_action_called_R = True
    #         shape_dim = _com_action_.get_shape()
    #         _com_action_ = tf.reshape(_com_action_, [-1, shape_dim[1] * shape_dim[2]])
    #         feedInput = tf.concat([_com_action_, p_w_latent_layer, e_p_latent_layer], axis=1)
    #         ca_fc1 = fc(feedInput, int((dim1 + dim2) / 2), 'fc1')
    #         ca_batchnorm1 = batch_norm(ca_fc1, self.is_train)
    #         ca_lrelu1 = leaky_relu(ca_batchnorm1)
    #         dim3 = (len(Vocab) + 3 * len(Emotions))
    #         ca_fc2_reshaped = tf.reshape(ca_lrelu1, [-1, dim3, Num_Agents])
    #         return tf.nn.sigmoid(ca_fc2_reshaped)
    #
    # def com_action2_real(self, _com_action_2, e_p_latent_layer2, p_w_latent_layer2, dim1, dim2):
    #     with tf.compat.v1.variable_scope('CAR2', reuse=self.com_action_called_R2):
    #         self.com_action_called_R2 = True
    #         shape_dim = _com_action_2.get_shape()
    #         _com_action_2 = tf.reshape(_com_action_2, [-1, shape_dim[1] * shape_dim[2]])
    #         feedInput = tf.concat([_com_action_2, p_w_latent_layer2, e_p_latent_layer2], axis=1)
    #         ca_fc1 = fc(feedInput, int((dim1 + dim2) / 2), 'fc1')
    #         ca_batchnorm1 = batch_norm(ca_fc1, self.is_train)
    #         ca_lrelu1 = leaky_relu(ca_batchnorm1)
    #         dim3 = (len(Vocab) + 3 * len(Emotions))
    #         ca_fc2_reshaped = tf.reshape(ca_lrelu1, [-1, dim3, Num_Agents])
    #         return tf.nn.sigmoid(ca_fc2_reshaped)
    #
    # def com_action3_real(self, _com_action_3, e_p_latent_layer3, p_w_latent_layer3, dim1, dim2):
    #     with tf.compat.v1.variable_scope('CAR3', reuse=self.com_action_called_R3):
    #         self.com_action_called_R3 = True
    #         shape_dim = _com_action_3.get_shape()
    #         _com_action_3 = tf.reshape(_com_action_3, [-1, shape_dim[1] * shape_dim[2]])
    #         feedInput = tf.concat([_com_action_3, p_w_latent_layer3, e_p_latent_layer3], axis=1)
    #         ca_fc1 = fc(feedInput, int((dim1 + dim2) / 2), 'fc1')
    #         ca_batchnorm1 = batch_norm(ca_fc1, self.is_train)
    #         ca_lrelu1 = leaky_relu(ca_batchnorm1)
    #         dim3 = (len(Vocab) + 3 * len(Emotions))
    #         ca_fc2_reshaped = tf.reshape(ca_lrelu1, [-1, dim3, Num_Agents])
    #         return tf.nn.sigmoid(ca_fc2_reshaped)


    def RNN(self, words_input, weights, biases):
        with tf.compat.v1.variable_scope('DEC', reuse=self.dec_called):
            self.dec_called = True
            # reshape to [1, n_input]
            x = tf.reshape(words_input, [-1, self.length_w, self.n_word_input_dim])

            # Generate a n_input-element sequence of inputs
            # (eg. [had] [a] [general] -> [20] [6] [33])
            #x = tf.split(x, self.n_word_input_dim, 1)

            # 1-layer LSTM with n_hidden units.
            lstm_cell = tf.keras.layers.LSTMCell(units=self.n_hidden_w, dropout=0.1, recurrent_dropout=0.1)
            # generate prediction
            layer = tf.keras.layers.RNN(lstm_cell) #, x, dtype=tf.float32
            outputs = layer(x)
            # output dim == latent output size w :100

            # there are n_input outputs but
            # we only want the last output
            return tf.matmul(outputs, weights['out']) + biases['out']


    def Entropy(self, Input):
        return tf.reduce_mean(-tf.reduce_sum(Input * tf.compat.v1.log(tf.add(Input, self.small_num)), axis=1))

    def _loss1(self, labels, logits):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        return tf.reduce_mean(loss)

    def _loss2(self, labels, logits):
        return tf.reduce_mean(tf.nn.l2_loss(labels - logits))

    def sess_close(self):
        self.sess.close()

    def test_one_patch(self, type_input, hp_input, angle_input, action_took_input, hf_value, words_input1,
                    words_input2, words_input3, emotion_input1, emotion_input2, emotion_input3, is_training):

        with tf.device(self.device):
            test_feed_dict = {self.type_input: type_input, self.hp_input: hp_input,
                               self.angle_input: angle_input, self.action_took_input: action_took_input,
                               self.hf_value: hf_value, self.words_input1: words_input1,
                               self.words_input2: words_input2, self.words_input3: words_input3,
                               self.emotion_input1: emotion_input1, self.emotion_input2: emotion_input2,
                               self.emotion_input3: emotion_input3, self.is_train: is_training}
            com_actions, actions, internal_mentality = self.sess.run([self._com_action,
                                                                      self._action, self.internal_mental_state],
                                                                     feed_dict=test_feed_dict)
        return com_actions, actions, internal_mentality
    def test_one_patch2(self, type_input, hp_input, angle_input, action_took_input, hf_value, words_input1,
                    words_input2, words_input3, emotion_input1, emotion_input2, emotion_input3, is_training):

        with tf.device(self.device):
            test_feed_dict = {self.type_input: type_input, self.hp_input: hp_input,
                               self.angle_input: angle_input, self.action_took_input: action_took_input,
                               self.hf_value: hf_value, self.words_input1: words_input1,
                               self.words_input2: words_input2, self.words_input3: words_input3,
                               self.emotion_input1: emotion_input1, self.emotion_input2: emotion_input2,
                               self.emotion_input3: emotion_input3, self.is_train: is_training}
            com_actions, actions, internal_mentality = self.sess.run([self._com_action2,
                                                                      self._action2, self.internal_mental_state],
                                                                     feed_dict=test_feed_dict)
        return com_actions, actions, internal_mentality
    def test_one_patch3(self, type_input, hp_input, angle_input, action_took_input, hf_value, words_input1,
                    words_input2, words_input3, emotion_input1, emotion_input2, emotion_input3, is_training):

        with tf.device(self.device):
            test_feed_dict = {self.type_input: type_input, self.hp_input: hp_input,
                               self.angle_input: angle_input, self.action_took_input: action_took_input,
                               self.hf_value: hf_value, self.words_input1: words_input1,
                               self.words_input2: words_input2, self.words_input3: words_input3,
                               self.emotion_input1: emotion_input1, self.emotion_input2: emotion_input2,
                               self.emotion_input3: emotion_input3, self.is_train: is_training}
            # com_actions, actions, internal_mentality, _com_action_r3 = self.sess.run([self._com_action3,
            #                                                           self._action3, self.internal_mental_state,
            #                                                           self._com_action3_r],
            #                                                          feed_dict=test_feed_dict)
            com_actions, actions, internal_mentality = self.sess.run([self._com_action3,
                                                                      self._action3, self.internal_mental_state],
                                                                     feed_dict=test_feed_dict)
        return com_actions, actions, internal_mentality #_com_action_r3

    def train_batch(self, gt_reward, type_input, hp_input, angle_input, action_took_input, hf_value, words_input1,
                    words_input2, words_input3, emotion_input1, emotion_input2, emotion_input3, is_training,
                    learning_rate_shifting, com_reward):

        with tf.device(self.device):
            train_feed_dict = {self.reward_got: gt_reward, self.type_input: type_input, self.hp_input: hp_input,
                               self.angle_input: angle_input, self.action_took_input: action_took_input,
                               self.hf_value: hf_value, self.words_input1: words_input1,
                               self.words_input2: words_input2, self.words_input3: words_input3,
                               self.emotion_input1: emotion_input1, self.emotion_input2: emotion_input2,
                               self.emotion_input3: emotion_input3, self.is_train: is_training,
                               self.learning_rate_shifting: learning_rate_shifting, self.com_reward: com_reward
                               }
            # ,
            # self.emotion_input1_prevreal: emotion_input1_prevreal,
            # self.emotion_input1_prevfake: emotion_input1_prevfake

            com_actions, actions, loss_val, _ = self.sess.run(
                [self._com_action, self._action, self.loss_op, self.train_op], feed_dict=train_feed_dict)

        return com_actions, actions, loss_val
    def train_batch2(self, gt_reward, type_input, hp_input, angle_input, action_took_input, hf_value, words_input1,
                    words_input2, words_input3, emotion_input1, emotion_input2, emotion_input3, is_training,
                    learning_rate_shifting, com_reward):

        with tf.device(self.device):
            train_feed_dict = {self.reward_got: gt_reward, self.type_input: type_input, self.hp_input: hp_input,
                               self.angle_input: angle_input, self.action_took_input: action_took_input,
                               self.hf_value: hf_value, self.words_input1: words_input1,
                               self.words_input2: words_input2, self.words_input3: words_input3,
                               self.emotion_input1: emotion_input1, self.emotion_input2: emotion_input2,
                               self.emotion_input3: emotion_input3, self.is_train: is_training,
                               self.learning_rate_shifting: learning_rate_shifting, self.com_reward: com_reward}

            # ,
            # self.emotion_input2_prevreal: emotion_input2_prevreal,
            # self.emotion_input2_prevfake: emotion_input2_prevfake

            com_actions, actions, loss_val, _ = self.sess.run(
                [self._com_action2, self._action2, self.loss_op2, self.train_op2], feed_dict=train_feed_dict)

        return com_actions, actions, loss_val
    def train_batch3(self, gt_reward, type_input, hp_input, angle_input, action_took_input, hf_value, words_input1,
                    words_input2, words_input3, emotion_input1, emotion_input2, emotion_input3, is_training,
                    learning_rate_shifting, com_reward):

        # , emotion_input3_prevreal, emotion_input3_prevfake


        with tf.device(self.device):
            train_feed_dict = {self.reward_got: gt_reward, self.type_input: type_input, self.hp_input: hp_input,
                               self.angle_input: angle_input, self.action_took_input: action_took_input,
                               self.hf_value: hf_value, self.words_input1: words_input1,
                               self.words_input2: words_input2, self.words_input3: words_input3,
                               self.emotion_input1: emotion_input1, self.emotion_input2: emotion_input2,
                               self.emotion_input3: emotion_input3, self.is_train: is_training,
                               self.learning_rate_shifting: learning_rate_shifting, self.com_reward: com_reward}

            # ,
            # self.emotion_input3_prevreal: emotion_input3_prevreal,
            # self.emotion_input3_prevfake: emotion_input3_prevfake

            #emotion_input3_prevreal: analise received com action 3 in the previous step
            #emotion_input3_prevfake: .. action 3 in the previous step

            com_actions, actions, loss_val, _ = self.sess.run(
                [self._com_action3, self._action3, self.loss_op3, self.train_op3], feed_dict=train_feed_dict)

        return com_actions, actions, loss_val




# with tf.Session() as sess:
#     with tf.device('/cpu:0'):
#         agent_net = agent_net()
#         sess.run(tf.global_variables_initializer())
#         agent_net.train(sess, train_samples)
#         dis_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dis')
#         gen_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gen')
#         saver = tf.train.Saver(dis_var_list + gen_var_list)
#         saver.save(sess, 'model/H_info_Gail')





