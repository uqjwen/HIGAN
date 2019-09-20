import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from data_loader import Data_Loader
import sys

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

		self.training 	= tf.placeholder(tf.bool)
		self.u_input 	= tf.placeholder(tf.int32, shape=[None,])
		self.i_input 	= tf.placeholder(tf.int32, shape=[None,])
		self.label 		= tf.placeholder(tf.int32, shape=[None,])

		self.u_text 	= tf.placeholder(tf.int32, shape=[None, self.t_num, self.maxlen])
		self.i_text 	= tf.placeholder(tf.int32, shape=[None, self.t_num, self.maxlen])
		self.u_embed 	= tf.get_variable('u_emb', [self.num_user, self.emb_size], initializer = xavier_initializer())
		self.i_embed 	= tf.get_variable('i_emb', [self.num_item, self.emb_size], initializer = xavier_initializer())
		self.w_embed 	= tf.get_variable('w_emb', [self.vocab_size, self.emb_size], initializer = xavier_initializer())


		self.u_latent 	= tf.nn.embedding_lookup(self.u_embed, self.u_input)
		self.i_latent 	= tf.nn.embedding_lookup(self.i_embed, self.i_input)


		self.ut_latent 	= tf.nn.embedding_lookup(self.w_embed, self.u_text)
		self.it_latent 	= tf.nn.embedding_lookup(self.w_embed, self.i_text)

		self.define_conv()


		self.ut_cnn = self.get_cnn(self.ut_latent)
		self.it_cnn = self.get_cnn(self.it_latent)


		self.ut_watt = self.get_word_level_att(self.ut_cnn, self.u_latent, 'user') #[?,6,100]
		self.it_watt = self.get_word_level_att(self.it_cnn, self.i_latent, 'item') #[?,6,100]


		self.get_doc_level_att(self.ut_watt, self.it_watt)

		# user_side = tf.concat([self.u_latent, self.doc_user[-1]], axis=-1)
		# item_side = tf.concat([self.i_latent, self.doc_item[-1]], axis=-1)

		# self.prediction = self.get_prediction(user_side, item_size)

		# self.loss = tf.reduce_mean(tf.square(self.prediction - tf.cast(self.label, tf.float32)))
		self.get_layer_loss()

		self.train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.layer_loss[-1])


	def define_conv(self):
		self.conv1 = tf.keras.layers.Conv1D(100, 3, padding='same')
		self.conv2 = tf.keras.layers.Conv1D(100, 3, padding='same')
		self.conv3 = tf.keras.layers.Conv1D(self.emb_size, 3, padding='same')
		self.dropout = tf.keras.layers.Dropout(0.5)


		self.w_att = tf.keras.layers.Dense(self.emb_size, activation = 'relu')

		self.ctt_dense = tf.keras.layers.Dense(1)

		self.word_user_alpha = None
		self.word_item_alpha = None
		self.doc_user_alpha = []
		self.doc_item_alpha = []


		self.mlp_layers = 3
		self.mlp_dense = []
		units = self.emb_size//2
		for i in range(self.mlp_layers):
			self.mlp_dense.append(tf.keras.layers.Dense(units, activation='relu'))
			units = units//2
		self.mlp_dense.append(tf.keras.layers.Dense(1))

		self.doc_user = []
		self.doc_item = []


	def get_cnn(self, latent):
		latent = tf.reshape(latent, [-1, self.maxlen, self.emb_size])
		conv1 = self.conv1(latent)
		conv1 = self.dropout(tf.nn.relu(conv1), training = self.training)
		conv2 = self.conv2(latent)
		conv2 = self.dropout(tf.nn.relu(conv2), training = self.training)
		conv3 = tf.concat([conv1,conv2], axis=-1)
		conv4 = self.conv3(conv3)
		ret = self.dropout(tf.nn.relu(conv4), training = self.training)


		# print(ret.shape)
		return ret

	def get_word_level_att(self, uit_cnn, ui_latent, name='user'):
		uit_cnn_rsh = tf.reshape(uit_cnn, [-1, self.t_num, self.maxlen, self.emb_size])
		trans = self.w_att(uit_cnn_rsh) #[?,6,60,100]
		trans = tf.reshape(trans, [-1, self.t_num, self.maxlen, self.emb_size])
		latent = tf.expand_dims(tf.expand_dims(ui_latent,1),1) #[?,1,1,100]
		alpha = tf.reduce_sum(trans*latent,axis=-1) #[?,6,60]
		alpha = tf.nn.softmax(alpha, axis=-1)
		if name == 'user':
			self.word_user_alpha = alpha
		else:
			self.word_item_alpha = alpha
		certainty = self.get_certainty(alpha)

		alpha = tf.expand_dims(alpha, axis=-1) #[?,6,60,1]

		hidden = tf.reduce_sum(alpha*uit_cnn_rsh, axis=2) #[?,6,100]

		print(certainty.shape, alpha.shape)

		return hidden*certainty

		
	def get_certainty(self,alpha): # ?,6,60
		# alpha_sort = tf.sort(alpha, axis=-1)
		alpha_mean = tf.reduce_mean(alpha, axis=-1, keepdims = True) # ?,6,1
		# alpha_sort = alpha
		upper_mask = alpha>alpha_mean
		upper_mask = tf.cast(upper_mask, tf.float32)
		lower_mask = 1.-upper_mask   # ?,6,60

		alpha_lower = tf.reduce_mean(alpha*lower_mask, axis=-1, keepdims = True) # ?,6,1
		alpha_upper = tf.reduce_mean(alpha*upper_mask, axis=-1, keepdims = True)

		# half = self.maxlen//2
		# alpha_lower = alpha_sort[:,:,:half]
		# alpha_lower = tf.reduce_mean(alpha_lower, axis=-1)
		# alpha_upper = alpha_sort[:,:,half:]
		# alpha_upper = tf.reduce_mean(alpha_upper, axis=-1)
		# alpha_mean = tf.reduce_mean(alpha,axis=-1)

		certainty = tf.nn.sigmoid((alpha_upper-alpha_mean)*(alpha_mean-alpha_lower))
		certainty = 2*certainty - 1
		# certainty = tf.expand_dims(certainty, axis=-1)
		return certainty

	def get_doc_level_att(self, u_watt, i_watt):
		docs_user = u_watt
		doc_user = tf.reduce_max(u_watt, axis=1, keepdims = True)
		self.doc_user.append(doc_user)

		docs_item = i_watt 
		doc_item = tf.reduce_max(i_watt, axis=1, keepdims = True)
		self.doc_item.append(doc_item)

		layers = 3
		for i in range(layers):
			i_temp = self.doc_level_att(self.doc_user[-1], docs_item, i, 'item')
			u_temp = self.doc_level_att(self.doc_item[-1], docs_user, i, 'user')


			i_pool = tf.concat([i_temp, self.doc_item[-1]], axis=1)
			i_l = tf.reduce_max(i_pool, axis=1, keepdims = True)
			self.doc_item.append(i_l)

			u_pool = tf.concat([u_temp, self.doc_user[-1]], axis=1)
			u_l = tf.reduce_max(u_pool, axis=1, keepdims = True)
			self.doc_user.append(u_l)

			# u_l_1 = u_l
			# i_l_1 = i_l
		# return u_l, i_l 


			# i_l = self.combine(i_l, )


	def doc_level_att(self, vec_1, vec_2, layer, name='user'):
		#vec1 ?,100
		#vec2 ?,6,100
		vec2 = tf.keras.layers.Dense(self.emb_size, activation = 'relu')(vec_2) 
		# vec1 = tf.expand_dims(vec_1, axis=1) # ?,1,100
		vec1 = vec_1
		alpha = tf.reduce_sum(vec1*vec2, axis=-1) #?,6
		alpha = tf.nn.softmax(alpha, axis=-1)
		alpha = tf.expand_dims(alpha, axis=-1) # ?,6,1
		if name == 'user':
			self.doc_user_alpha.append(alpha)
		else:
			self.doc_item_alpha.append(alpha)
			
		return tf.reduce_sum(alpha*vec_2, axis=1, keepdims = True) # ?,1,100

	def get_prediction(self,vec1, vec2):

		hidden = vec1*vec2
		# temp_layer = [hidden]
		for mlp_layer in self.mlp_dense:
			hidden = mlp_layer(hidden)
		return hidden


	def get_layer_loss(self):
		self.layer_loss = []
		for i in range(len(self.doc_user)):
			u_side = tf.concat([self.u_latent, tf.squeeze(self.doc_user[i],axis=1)], axis=-1)
			i_side = tf.concat([self.i_latent, tf.squeeze(self.doc_item[i],axis=1)], axis=-1)
			prediction = self.get_prediction(u_side, i_side)
			loss = tf.reduce_mean(tf.square(prediction - tf.cast(self.label, tf.float32)))
			self.layer_loss.append(loss)



if __name__ == '__main__':
	filename = 'Musical_Instruments_5.json'


	flags = tf.flags.FLAGS 	
	tf.flags.DEFINE_string('filename',filename,'name of file')
	tf.flags.DEFINE_integer('batch_size',2,'batch size')
	tf.flags.DEFINE_integer('emb_size',100, 'embedding size')
	tf.flags.DEFINE_integer('num_class', 5, "num of classes")
	# tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
	flags(sys.argv)

	data_loader = Data_Loader(flags)

	model = Model(flags, data_loader)

