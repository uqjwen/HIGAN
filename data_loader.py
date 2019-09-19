import numpy as np 
import pickle
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import sys

class Data_Loader():
	def __init__(self, flags):
		data = pickle.load(open('./data/'+flags.filename.split('.')[0]+'.pkl','rb'))
		# print(len(data))
		vec_texts 	= data['vec_texts']
		vocab 		= data['vocab']
		vec_uit 	= data['vec_uit']
		vec_u_text 	= data['vec_u_text']
		vec_i_text 	= data['vec_i_text']
		user2idx 	= data['user2idx']
		item2idx 	= data['item2idx']
		self.batch_size = flags.batch_size
		self.num_user 	= len(user2idx)
		self.num_item 	= len(item2idx)
		self.vocab_size = len(vocab)+1
		self.data_size 	= len(vec_uit)
		self.emb_size 	= flags.emb_size

		self.d_num 	= 6
		self.maxlen = 60
		self.vec_texts 	= pad_sequences(vec_texts, maxlen = self.maxlen)
		self.vec_u_text = pad_sequences(vec_u_text, maxlen = self.d_num)
		self.vec_i_text = pad_sequences(vec_i_text, maxlen = self.d_num)

		self.vec_uit = np.array(vec_uit)

		self.vec_uit = np.random.permutation(self.vec_uit)


		self.train_size = int(self.data_size*0.9)

		self.train_uit = self.vec_uit[:self.train_size]
		self.text_uit = self.vec_uit[self.train_size:]

		self.pointer = 0
	def next_batch(self):
		begin 	= self.pointer*self.batch_size
		end		= (self.pointer+1)*self.batch_size
		self.pointer+=1
		if end >= self.train_size:
			end = self.train_size
			self.pointer = 0

		labels = self.train_uit[begin:end][:,3]
		users = self.train_uit[begin:end][:,0]
		items = self.train_uit[begin:end][:,1]
		texts = self.train_uit[begin:end][:,2]

		utexts = self.vec_u_text[users]
		itexts = self.vec_i_text[items]
		u_texts = self.vec_texts[utexts]
		i_texts = self.vec_texts[itexts]

		print(users)
		print(utexts)
		print(itexts)
		# print(u_texts)
		# print(i_texts)
		return users, items, labels, u_texts, i_texts
	def reset_pointer(self):
		self.pointer = 0


if __name__ == '__main__':
	filename = 'Musical_Instruments_5.json'


	flags = tf.flags.FLAGS 	
	tf.flags.DEFINE_string('filename',filename,'name of file')
	tf.flags.DEFINE_integer('batch_size',2,'batch size')
	tf.flags.DEFINE_integer('emb_size',100, 'embedding size')
	# tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
	flags(sys.argv)

	data_loader = Data_Loader(flags)

	data_loader.next_batch()
	data_loader.next_batch()