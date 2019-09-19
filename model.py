import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

class Model(object):
	def __init__(self, flags, data_loader):
		self.vocab_size = data_loader.vocab_size
		self.num_user 	= data_loader.num_user
		self.num_item 	= data_loader.num_item
		self.num_class 	= flags.num_class
		self.emb_size 	= flags.emb_size
		self.batch_size = flags.batch_size
		self.t_num 		= data_loader.t_num
		self.maxlen 	= data_loader.maxlen

		self.training = tf.placeholder(tf.bool)
		self.u_input = tf.placeholder(tf.int32, shape=[None, 1])
		self.i_input = tf.placeholder(tf.int32, shape=[None, 1])

		self.u_text = tf.placeholder(tf.int32, shape=[None, self.t_num, self.maxlen])
		self.i_text = tf.placeholder(tf.int32, shape=[None, self.t_num, self.maxlen])
		self.u_embed = tf.get_variable('u_emb', [self.num_user, self.emb_size], initializer = xavier_initializer())
		self.i_embed = tf.get_variable('i_emb', [self.num_item, self.emb_size], initializer = xavier_initializer())
		self.w_embed = tf.get_variable('w_emb', [self.vocab_size, self.emb_size], initializer = xavier_initializer())


	def define_conv(self):
		self.conv1 = tf.keras.layers.Conv1D(100, 3, padding='same')
		self.conv2 = tf.keras.layers.Conv1D(100, 3, padding='same')
		self.conv3 = tf.keras.layers.Conv1D(self.emb_size, 3, padding='same')

		self.dropout = tf.keras.layers.Dropout(self.drop)