import sys
import os
import numpy as np 
import json 
import nltk
import pickle
stopwords = nltk.corpus.stopwords.words('english')

def readfile(filename):
	f = open(filename, encoding='utf-8')
	data = []
	u_dict = {}
	i_dict = {}
	texts = []
	uit = []

	for line in f.readlines():
		line = json.loads(line)
		user = line['reviewerID']
		item = line['asin']
		text = line['reviewText']
		unixtime = line['unixReviewTime']
		rating = line['overall']
		texts.append(text)
		uit.append([user,item, rating,unixtime])

		index = len(texts)-1
		if user not in u_dict:
			u_dict[user] = [index]
		else:
			u_dict[user].append(index)

		if item not in i_dict:
			i_dict[item] = [index]
		else:
			i_dict[item].append(index)


	user2idx = {user:i for i,user in enumerate(u_dict.keys())}
	item2idx = {item:i for i,item in enumerate(i_dict.keys())}



	length = len(texts)
	train_len = int(0.8*length)
	val_len = int(0.1*length)
	test_len = int(0.1*length)


	train_uit = [[user2idx[tp[0]],item2idx[tp[1]], tp[2], tp[3] ]    for tp in uit[:train_len]]
	val_uit = [[user2idx[tp[0]],item2idx[tp[1]], tp[2], tp[3] ]    for tp in uit[train_len:train_len+val_len]]
	test_uit = [[user2idx[tp[0]],item2idx[tp[1]], tp[2], tp[3] ]    for tp in uit[train_len+val_len:]]
	np.savetxt('TrainInteraction.out', train_uit, fmt="%d", delimiter = '\t')
	np.savetxt('ValInteraction.out', val_uit, fmt="%d", delimiter = '\t')
	np.savetxt('TestInteraction.out', test_uit, fmt="%d", delimiter = '\t')

	vocab = get_vocab_freq(texts)
	
	word2idx = [[word,str(i)] for i,word in enumerate(vocab)]

	word_dict = {word:i for i,word in enumerate(vocab)}

	np.savetxt('WordDict.out', word2idx, fmt='%s', delimiter='\t')


	ui_review(u_dict, user2idx, word_dict, texts, train_len, 'UserReviews.out')

	ui_review(i_dict, item2idx, word_dict, texts, train_len, 'ItemReviews.out')

	f.close()


def ui_review(ui_dict, ui2idx,word2idx, texts, train_len, filename):
	fr = open(filename, 'w')
	for user in ui_dict.keys():
		reviews = ''
		for text_id in ui_dict[user]:
			if text_id>=train_len:
				continue
			else:
				tokens = nltk.word_tokenize(texts[text_id])
				tokens = [token for token in tokens if token in word2idx]
				reviews += ' '.join(tokens)+' '

		fr.write(str(ui2idx[user])+'\t'+reviews)
		fr.write('\n')
	fr.close()




def get_vocab_freq(corpus, vocab_size=20000):
	freq_dict = {}
	for text in corpus:
		tokens = nltk.word_tokenize(text)
		for token in tokens:
			freq_dict[token] = freq_dict.get(token, 0)+1
	tuples = [[v[1],v[0]] for v in freq_dict.items()]
	tuples = sorted(tuples)[::-1]
	vocab = []
	for tp in tuples:
		word = tp[1]
		if word not in stopwords:
			vocab.append(word)
			if len(vocab)>20000:
				break
	return vocab




	# return data
if __name__ == '__main__':
	readfile('Musical_Instruments_5.json')
