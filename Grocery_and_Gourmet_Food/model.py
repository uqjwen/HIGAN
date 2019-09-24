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
		self.data_size  = data_loader.data_size+1
		self.vec_texts 	= data_loader.vec_texts

		self.u_input 	= tf.placeholder(tf.int32, shape=[None,])
		self.i_input 	= tf.placeholder(tf.int32, shape=[None,])
		self.text 	 	= tf.placeholder(tf.int32, shape=[None,])
		self.keep_prob  = tf.placeholder(tf.float32)
		self.utext 		= tf.placeholder(tf.int32, shape=[None, self.t_num])
		self.itext 		= tf.placeholder(tf.int32, shape=[None, self.t_num])
		self.label 		= tf.placeholder(tf.float32, shape=[None,1])




		self.u_embed 	= tf.get_variable('u_emb', [self.num_user, self.emb_size], initializer = xavier_initializer())
		self.i_embed 	= tf.get_variable('i_emb', [self.num_item, self.emb_size], initializer = xavier_initializer())
		self.w_embed 	= tf.get_variable('w_emb', [self.vocab_size, self.emb_size], initializer = xavier_initializer())
		self.s_embed 	= tf.get_variable('s_emb', [self.data_size, self.emb_size], initializer = xavier_initializer())

		self.define_var()


		self.udocs = tf.nn.embedding_lookup(self.vec_texts, self.utext)
		self.idocs = tf.nn.embedding_lookup(self.vec_texts, self.itext) # ?,8,60

		self.uw_emb = tf.nn.embedding_lookup(self.w_embed, self.udocs)
		self.iw_emb = tf.nn.embedding_lookup(self.w_embed, self.idocs) # ?,8,60,100

		self.u_latent 	= tf.nn.embedding_lookup(self.u_embed, self.u_input)
		self.i_latent 	= tf.nn.embedding_lookup(self.i_embed, self.i_input)



		# text_words = tf.nn.embedding_lookup(self.vec_texts, self.text)
		# text_latent = tf.nn.embedding_lookup(self.w_embed, text_words)

		self.u_cnn = self.get_cnn(self.uw_emb)
		self.i_cnn = self.get_cnn(self.iw_emb)

		print(self.uw_emb.shape, self.u_cnn.shape)

		self.uw_att = self.get_word_level_att(self.u_cnn, self.u_latent, self.i_latent, 'user')
		self.iw_att = self.get_word_level_att(self.i_cnn, self.u_latent, self.i_latent, 'item')




		self.get_doc_level_att(self.uw_att, self.iw_att)


		self.get_layer_loss()


		# loss = tf.reduce_mean(tf.square(score - tf.cast(self.label, tf.float32)))
		# self.mae = tf.reduce_mean(tf.abs( - tf.cast(self.label, tf.float32)))

		# self.loss = loss

		self.train_op = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(self.layer_loss[-1])


	def get_cnn(self, latent):
		latent = tf.reshape(latent, [-1, self.maxlen, self.emb_size])
		conv1 = self.conv1(latent)
		conv2 = self.conv2(latent)

		# conv1 = tf.contrib.layers.batch_norm(tf.expand_dims(conv1,-1))
		# conv1 = tf.squeeze(conv1,-1)

		# conv2 = tf.contrib.layers.batch_norm(tf.expand_dims(conv2,-1))
		# conv2 = tf.squeeze(conv2,-1)


		hidden = tf.nn.relu(tf.concat([conv1,conv2], axis=-1))
		hidden = tf.nn.dropout(hidden, self.keep_prob)

		conv3 = tf.nn.relu(self.conv3(hidden))

		# conv3 = tf.contrib.layers.batch_norm(tf.expand_dims(conv3,-1))
		# conv3 = tf.squeeze(conv3,-1)



		conv3 = tf.nn.dropout(conv3, self.keep_prob)

		return conv3


	def define_var(self):
		self.conv1 = tf.keras.layers.Conv1D(128, 3, padding='same')
		self.conv2 = tf.keras.layers.Conv1D(128, 5, padding='same')
		self.conv3 = tf.keras.layers.Conv1D(self.emb_size, 3, padding='same')
		# self.dropout = tf.keras.layers.Dropout(0.5)


		self.w_att = tf.keras.layers.Dense(self.emb_size*2, activation = 'tanh')

		self.d_att = tf.keras.layers.Dense(1,'tanh')

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
		# self.prediction = []
		self.layer_mae = []
		self.layer_loss = []

	def get_mlp_score(self,vec):
		size = vec.shape.as_list()[-1]//2
		layer = 2
		for i in range(layer):
			hidden = tf.keras.layers.Dense(size,activation='relu')(vec)
			vec = hidden
			size = size//2
		score = tf.keras.layers.Dense(1)(vec)
		return score

	def get_itr_score(self,vec1,vec2):
		vec = tf.concat([vec1,vec2,vec1*vec2])


	def get_word_level_att(self, uit_cnn, u_latent, i_latent, name='user'):

		uit_cnn_rsh = tf.reshape(uit_cnn, [-1, self.t_num, self.maxlen, self.emb_size])



		trans = self.w_att(uit_cnn_rsh) #[?,8,60,200]

		ui_latent = tf.concat([u_latent, i_latent], axis=-1)
		latent = tf.expand_dims(tf.expand_dims(ui_latent,1),1) #[?,1,1,200]
		alpha = tf.reduce_sum(trans*latent,axis=-1) #[?,8,60]
		# if name == 'user':
		# 	self.udebug = alpha
		# else:
		# 	self.idebug = alpha

		alpha = tf.nn.softmax(alpha, axis=-1)
		if name == 'user':
			self.word_user_alpha = alpha
		else:
			self.word_item_alpha = alpha
		certainty = self.get_certainty(alpha)
		self.certainty = certainty

		alpha = tf.expand_dims(alpha, axis=-1) #[?,8,60,1]

		hidden = tf.reduce_sum(alpha*uit_cnn_rsh, axis=2) #[?,8,100]

		print(certainty.shape, alpha.shape)

		return hidden#*certainty

	def get_certainty(self,alpha): # ?,6,60
		# alpha_sort = tf.sort(alpha, axis=-1)
		alpha_mean = tf.reduce_mean(alpha, axis=-1, keepdims = True) # ?,6,1
		# alpha_sort = alpha
		upper_mask = alpha>alpha_mean
		upper_mask = tf.cast(upper_mask, tf.float32)
		lower_mask = 1.-upper_mask   # ?,6,60

		alpha_lower = tf.reduce_mean(alpha*lower_mask, axis=-1, keepdims = True) # ?,6,1
		alpha_upper = tf.reduce_mean(alpha*upper_mask, axis=-1, keepdims = True)

		certainty = tf.nn.sigmoid((alpha_upper-alpha_mean)*(alpha_mean-alpha_lower))
		certainty = 2*certainty - 1
		# certainty = tf.expand_dims(certainty, axis=-1)
		return certainty

	def get_initial_vec(self,vec_1, vec_2):
		#mat_1  #?,6,100
		#mat_2  #?,8,100
		rows = vec_1.shape[1]
		cols = vec_2.shape[1]
		mat_1 = tf.expand_dims(vec_1,2) # ?,6,1,100
		mat_2 = tf.expand_dims(vec_2,1) # ?,1,8,100
		# mat_1 = tf.tile(mat_1, [1,1,cols,1]) # ?,6,8,100
		# mat_2 = tf.tile(mat_2, [1,rows,1,1])  # ?,6,8,100

		mat = tf.reduce_mean(tf.square(mat_1 - mat_2), axis=-1) #?,6,8

		alpha1 = -tf.reduce_min(mat, axis=-1)*10 #?,6

		alpha2 = -tf.reduce_min(mat, axis=1)*10 #?,8

		alpha1 = tf.nn.softmax(alpha1)
		alpha1 = tf.expand_dims(alpha1, axis=-1) #?,6,1

		alpha2 = tf.nn.softmax(alpha2)
		alpha2 = tf.expand_dims(alpha2, axis=-1) # ?,8,1


		vec1 = tf.reduce_sum(alpha1*vec_1, axis=1, keepdims = True) # ?,100

		vec2 = tf.reduce_sum(alpha2*vec_2, axis=1, keepdims = True)

		return vec1, vec2
		

	def get_doc_level_att(self, u_watt, i_watt):
		doc_user = tf.reduce_mean(u_watt, axis=1, keepdims = True)
		# doc_user, doc_item = self.get_initial_vec(u_watt, i_watt)
		docs_user = u_watt
		# doc_user = u_watt[:,0:1,:]
		self.doc_user.append(doc_user)

		docs_item = i_watt 
		# doc_item = i_watt[:,0:1,:]
		doc_item = tf.reduce_mean(i_watt, axis=1, keepdims = True)
		self.doc_item.append(doc_item)

		self.u_watt = u_watt 
		self.i_watt = i_watt 

		layers = 3
		for i in range(layers):
			i_temp = self.doc_level_att(self.doc_user[-1], docs_item, i, 'item')
			u_temp = self.doc_level_att(self.doc_item[-1], docs_user, i, 'user')


			# i_pool = tf.concat([i_temp, self.doc_item[-1]], axis=1)
			# i_temp = tf.reduce_max(i_pool, axis=1, keepdims = True)
			self.doc_item.append(i_temp)
			# self.doc_item.append(doc_item)

			# u_pool = tf.concat([u_temp, self.doc_user[-1]], axis=1)
			# u_temp = tf.reduce_max(u_pool, axis=1, keepdims = True)
			self.doc_user.append(u_temp)
			# self.doc_user.append(doc_user)



	def doc_level_att(self, vec_1, vec_2, layer, name='user'):
		#vec1 ?,1,100
		#vec2 ?,6,100
		vec2 = vec_2
		vec1 = vec_1


		# alpha_1 = self.d_att(vec1-vec2)  # ?,6,100
		# alpha_2 = tf.nn.softmax(alpha_1, axis=1)
		# if name == 'user':
		# 	self.doc_user_alpha.append(tf.squeeze(alpha_2, axis=-1))
		# else:
		# 	self.doc_item_alpha.append(tf.squeeze(alpha_2, axis=-1))

		# alpha_0 = tf.reduce_sum(vec1*vec2, axis=-1) #?,6
		# alpha_1 = tf.nn.softmax(alpha_0, axis=-1)
		# alpha_2 = tf.expand_dims(alpha_1, axis=-1) # ?,6,1
		# if name == 'user':
		# 	self.doc_user_alpha.append(alpha_1)
		# else:
		# 	self.doc_item_alpha.append(alpha_1)
			
		# return tf.reduce_sum(alpha_2*vec_2, axis=1, keepdims = True) # ?,1,100

		dist = tf.reduce_mean(tf.square(vec_1 - vec_2), axis=-1)*(layer+1)*10
		dist = -dist
		if layer == 0:
			self.vec_1 = vec_1
			self.vec_2 = vec_2
		alpha_1 = tf.nn.softmax(dist, axis=-1) # ?,6
		alpha_2 = tf.expand_dims(alpha_1, axis=-1)
		if name == 'user':
			self.doc_user_alpha.append(alpha_1)
		else:
			self.doc_item_alpha.append(alpha_1)



		return tf.reduce_sum(alpha_2*vec_2, axis=1, keepdims = True)


	def get_prediction(self,vec1, vec2):

		# hidden = vec1*vec2
		hidden = tf.concat([vec1, vec2, vec1*vec2], axis=-1)
		# temp_layer = [hidden]
		for mlp_layer in self.mlp_dense:
			hidden = mlp_layer(hidden)
		return hidden


	def get_layer_loss(self):
		for i in range(len(self.doc_user)):

			u_side = self.u_latent+tf.squeeze(self.doc_user[i],axis=1)
			i_side = self.i_latent+tf.squeeze(self.doc_item[i],axis=1)
			# u_side = tf.squeeze(self.doc_user[i], axis=1)
			# i_side = tf.squeeze(self.doc_item[i], axis=1)
			prediction = self.get_prediction(u_side, i_side)
			# self.prediction.append(prediction)
			loss = tf.reduce_mean(tf.square(prediction - tf.cast(self.label, tf.float32)))
			mae = tf.reduce_mean(tf.abs(prediction - tf.cast(self.label, tf.float32)))
			self.layer_loss.append(loss)

			self.layer_mae.append(mae)



if __name__ == '__main__':
	prefix = '/home/wenjh/aHIGAN/Grocery_and_Gourmet_Food/'
	filename = prefix+'Grocery_and_Gourmet_Food_5.json'


	flags = tf.flags.FLAGS 	
	tf.flags.DEFINE_string('filename',filename,'name of file')
	tf.flags.DEFINE_integer('batch_size',2,'batch size')
	tf.flags.DEFINE_integer('emb_size',100, 'embedding size')
	tf.flags.DEFINE_integer('num_class', 5, "num of classes")
	# tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
	flags(sys.argv)

	data_loader = Data_Loader(flags)

	model = Model(flags, data_loader)
