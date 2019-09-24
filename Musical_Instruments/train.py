import os 
import tensorflow as tf 
import numpy as np 
import model
from data_loader import Data_Loader 
import sys
import time 
from model import Model
from utils import visual
def train(sess, model, data_loader, flags):
	saver = tf.train.Saver(max_to_keep=1)

	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" [*] loading parameters success !!!")
	else:
		print(" [!] loading parameters failed  ...")

	best_rmse = 50
	fr = open(flags.ckpt_dir+'/loss.txt', 'a')
	mean_loss = []
	for i in range(flags.epoch):
		data_loader.reset_pointer()
		batches = data_loader.train_size // flags.batch_size+1
		for b in range(batches):
			u_input, i_input, label, utext, itext, text = data_loader.next_batch()

			feed_dict = {model.u_input: u_input,
						model.i_input: i_input,
						model.label: label,
						model.utext: utext,
						model.itext: itext,
						model.text:text,
						model.keep_prob: 0.5}
			_,loss = sess.run([model.train_op, model.layer_loss[-1]], feed_dict = feed_dict)

			# print(pred)
			# print(label)
			sys.stdout.write('\repoch:{}, batch:{}/{}, loss:{}'.format(i,b,batches,loss))
			sys.stdout.flush()
			mean_loss.append(loss)
			trained_batches = i*batches+b 
			if trained_batches!=0 and trained_batches%100 == 0:
				rmse = validation(sess, model, data_loader)
				print('\n',rmse[-1])
				loss = np.mean(mean_loss)
				print(loss)
				mean_loss = []
				# rmse = eval_by_batch(sess, model, data_loader)
				loss = round(loss,5)
				# rmse = round(rmse)
				rmse = [round(item,5) for item in rmse]

				fr.write(str(loss)+'\t'+'\t'.join(map(str, rmse)))
				fr.write('\n')
				if rmse[-1] < best_rmse:
					best_rmse = rmse[-1]
					print('saving....')
					saver.save(sess, flags.ckpt_dir+'/model.ckpt', global_step = trained_batches)



def validation(sess, model, data_loader):
	u_input, i_input, label, utext, itext, text = data_loader.eval()
	feed_dict = {model.u_input: u_input,
				model.i_input: i_input,
				model.label: label,
				model.utext: utext,
				model.itext: itext,
				model.text: text,
				model.keep_prob: 1.0}
	mae = sess.run(model.layer_mae, feed_dict = feed_dict)
	# print(loss)
	# loss = np.sqrt(loss)
	return mae

def evaluation(sess, model, data_loader, flags):
	saver = tf.train.Saver(max_to_keep=1)

	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" [*] loading parameters success !!!")
	else:
		print(" [!] loading parameters failed  ...")
		return 


	u_input, i_input, label, utext, itext, text = data_loader.eval()
	feed_dict = {model.u_input: u_input,
				model.i_input: i_input,
				model.label: label,
				model.utext: utext,
				model.itext: itext,
				model.text: text,
				model.keep_prob: 1.0}
	mae, docitem, docuser = sess.run([model.layer_mae, model.doc_item, model.doc_user], feed_dict = feed_dict)
	np.save('docitem.npy', docitem)
	np.save('docuser.npy', docuser)
	# print(loss)
	# loss = np.sqrt(loss)
	# print(uwatt)
	return mae

# def eval_by_batch(sess, model, data_loader):
# 	eval_data = data_loader.eval()
# 	test_size = len(eval_data[0])
# 	batches = test_size//data_loader.batch_size+1
# 	loss = []
# 	for i in range(batches):
# 		batch_data = []
# 		begin = i*data_loader.batch_size
# 		end = (i+1)*data_loader.batch_size
# 		end = min(end,test_size)
# 		for sub_data in eval_data:
# 			batch_data.append(sub_data[begin:end])
# 		feed_dict = {model.u_input: batch_data[0],
# 					model.i_input: batch_data[1],
# 					model.label: batch_data[2],
# 					model.u_text: batch_data[3],
# 					model.i_text: batch_data[4],
# 					model.training: False}
# 		batch_loss = sess.run(model.layer_loss, feed_dict = feed_dict)
# 		# print(batch_loss)
# 		loss.append(batch_loss)
# 	loss = np.array(loss)

# 	return np.mean(loss, axis=0)


def visualization(sess, model, data_loader, filename):
	saver = tf.train.Saver(max_to_keep=1)
	ckpt = tf.train.get_checkpoint_state(flags.ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(" [*] loading parameters success !!!")
	else:
		print(" [!] loading parameters failed  ...")
		return
	# user,item,label, utexts, itexts, text= data_loader.sample_point()
	user,item,label, utexts, itexts, text= data_loader.find_a_user()

	feed_dict = {model.u_input: user,
				model.i_input: item,
				model.label: label,
				model.utext: utexts,
				model.itext: itexts,
				model.text:text,
				model.keep_prob: 1.0}
	res = sess.run([model.word_user_alpha, model.word_item_alpha, model.doc_user_alpha, model.doc_item_alpha], feed_dict = feed_dict)


	u_texts = data_loader.vec_texts[utexts]
	i_texts = data_loader.vec_texts[itexts]
	res[2] = np.array(res[2]).transpose(1,0,2)
	res[3] = np.array(res[3]).transpose(1,0,2)

	for i in range(len(user)):
		uit = [user[i], item[i], label[i]]
		print(uit)
		res_trans = []
		for r in res:
			res_trans.append(r[i])
		visual(res_trans,uit, data_loader,utexts[i], itexts[i], u_texts[i], i_texts[i], filename)








if __name__ == '__main__':
	prefix = '/home/wenjh/aHIGAN/Musical_Instruments/'
	filename = prefix+'Musical_Instruments_5.json'


	flags = tf.flags.FLAGS 	
	tf.flags.DEFINE_string('filename',filename,'name of file')
	tf.flags.DEFINE_integer('batch_size',64,'batch size')
	tf.flags.DEFINE_integer('emb_size',100, 'embedding size')
	tf.flags.DEFINE_integer('num_class', 5, "num of classes")
	tf.flags.DEFINE_integer('epoch', 40, 'epochs for training')
	tf.flags.DEFINE_string('ckpt_dir',filename.split('.')[0], 'directory of checkpoint')
	tf.flags.DEFINE_string('train_test', 'train', 'training or test')
	# tf.flags.DEFINE_string('base_model', 'att_cnn', 'base model')
	flags(sys.argv)

	data_loader = Data_Loader(flags)

	model = Model(flags, data_loader)

	sess = tf.Session()

	sess.run(tf.global_variables_initializer())
	sess.run(model.w_embed.assign(data_loader.w_embed))

	if not os.path.exists(flags.ckpt_dir):
		os.makedirs(flags.ckpt_dir)

	if flags.train_test == 'train':
		train(sess, model, data_loader, flags)
	elif flags.train_test == 'visual':
		visualization(sess, model, data_loader, filename)
	elif flags.train_test == 'eval':
		evaluation(sess, model, data_loader, flags)
	# else:
	# 	test(sess, model, data_loader, flags)