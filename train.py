import os 
import tensorflow as tf 
import numpy as np 
import model
from data_loader import Data_Loader 
import sys
import time 
from model import Model

def train(sess, model, data_loader, flags):
	saver = tf.train.Saver(max_to_keep=1)

	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" [*] loading parameters success !!!")
	else:
		print(" [!] loading parameters failed  ...")

	best_rmse = 10
	fr = open(flags.ckpt_dir+'/loss.txt', 'a')
	for i in range(flags.epoch):
		data_loader.reset_pointer()
		batches = data_loader.train_size // flags.batch_size+1
		for b in range(batches):
			u_input, i_input, label, u_text, i_text = data_loader.next_batch()

			feed_dict = {model.u_input: u_input,
						model.i_input: i_input,
						model.label: label,
						model.u_text: u_text,
						model.i_text: i_text,
						model.training: True}
			_,loss = sess.run([model.train_op, model.loss], feed_dict = feed_dict)
			sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'.format(i,b,batches,loss))
			sys.stdout.flush()
			trained_batches = i*batches+b 
			if trained_batches!=0 and trained_batches%100 == 0:
				rmse = eval(sess, model, data_loader)
				loss = round(loss,5)
				rmse = round(rmse,5)
				fr.write(str(loss)+'\t'+str(rmse))
				fr.write('\n')
				if rmse < best_rmse:
					best_rmse = rmse
					save.save(sess, flags.ckpt_dir+'/model.ckpt', global_step = trained_batches)



def eval(sess, model, data_loader):
	u_input, i_input, label, u_text, i_text = data_loader.eval()
	feed_dict = {model.u_input: u_input,
				model.i_input: i_input,
				model.label: label,
				model.u_text: u_text,
				model.i_text: i_text,
				model.training: False}
	loss = sess.run(model.loss, feed_dict = feed_dict)
	loss = np.sqrt(loss)
	return loss


if __name__ == '__main__':
	filename = 'Musical_Instruments_5.json'


	flags = tf.flags.FLAGS 	
	tf.flags.DEFINE_string('filename',filename,'name of file')
	tf.flags.DEFINE_integer('batch_size',64,'batch size')
	tf.flags.DEFINE_integer('emb_size',100, 'embedding size')
	tf.flags.DEFINE_integer('num_class', 5, "num of classes")
	tf.flags.DEFINE_integer('epoch', 30, 'epochs for training')
	tf.flags.DEFINE_string('ckpt_dir',filename.split('.')[0], 'directory of checkpoint')
	tf.flags.DEFINE_string('train_test', 'train', 'training or test')
	# tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
	flags(sys.argv)

	data_loader = Data_Loader(flags)

	model = Model(flags, data_loader)

	sess = tf.Session()
	tf.set_random_seed(1234)
	np.random.seed(1234)

	sess.run(tf.global_variables_initializer())
	sess.run(model.w_embed.assign(data_loader.w_embed))

	if not os.path.exists(flags.ckpt_dir):
		os.makedirs(flags.ckpt_dir)

	if flags.train_test == 'train':
		train(sess, model, data_loader, flags)
	# else:
	# 	test(sess, model, data_loader, flags)