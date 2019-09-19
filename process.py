import numpy as np 
import json 
import nltk
import nltk
import pickle
stopwords = nltk.corpus.stopwords.words('english')

def readfile(filename):
	f = open(filename, encoding='utf-8')
	data = []
	for line in f.readlines():
		line = json.loads(line)
		data.append(line)
	f.close()
	return data
def filter_ui(data, u_min, i_min):
	u_dict = {}
	i_dict = {}
	for line in data:
		user = line['reviewerID']
		item = line['asin']
		if user not in u_dict:
			u_dict[user] = [item]
		else:
			u_dict[user].append(item)
		if item not in i_dict:
			i_dict[item] = [user]
		else:
			i_dict[item].append(user)


	flag = 0
	while flag == 0:
		flag = 1
		for user in u_dict.keys():
			if len(u_dict[user]) < u_min:
				for item in u_dict[user]:
					i_dict[item].remove(user)
				u_dict.pop(user)
				flag = 0
		for item in i_dict.keys():
			if len(i_dict[item]) < i_min:
				for user in i_dict[item]:
					u_dict[user].remove(item)
				i_dict.pop(item)
				flag = 0
	return u_dict,i_dict

def data_after_filter(data, u_dict, i_dict):
	new_data = []
	for line in data:
		if line['asin'] in i_dict and line['reviewerID'] in u_dict:
			new_data.append(line)
	return new_data


def build_ui_vocab(data):
	ut_dict = {}
	u_dict = {}
	u_list = []

	it_dict = {}
	i_dict = {}
	i_list = []
	for i,line in enumerate(data):
		user = line['reviewerID']
		item = line['asin']
		idx = i+1


		if user not in u_dict:
			u_dict[user] = [idx]
		else:u_dict[user].append(idx)

		if item not in i_dict:
			i_dict[item] = [idx]
		else:i_dict[item].append(idx)
	user2idx = {user:i for i,user in enumerate(u_dict.keys())}
	idx2user = {i:user for i,user in enumerate(u_dict.keys())}
	item2idx = {item:i for i,item in enumerate(i_dict.keys())}
	idx2item = {i:item for i,item in enumerate(i_dict.keys())}
	vec_u_text = vectorize_ui_text(idx2user, u_dict)
	vec_i_text = vectorize_ui_text(idx2item, i_dict)

	vec_uit = []
	for i,line in enumerate(data):
		user = line['reviewerID']
		item = line['asin']
		rate = int(line['overall'])
		text_idx = i+1
		user_idx = user2idx[user]
		item_idx = item2idx[item]

		vec_uit.append([user_idx, item_idx, text_idx, rate])
	return vec_uit, vec_u_text, vec_i_text, user2idx, item2idx

def vectorize_ui_text(idx2ui, ui_dict):
	vec_ui_text = []
	for idx in range(len(ui_dict)):
		ui = idx2ui[idx]
		text_idx = ui_dict[ui]
		vec_ui_text.append(np.random.permutation(text_idx))
	return vec_ui_text


def build_vocab(data):
	texts = []
	# u_texts = {}
	# i_texts = {}
	from utils import clean_str
	for line in data:
		text = line['reviewText']
		text = clean_str(text.lower())
		texts.append(text)
		
	vocab = get_vocab_freq(texts)

	word2idx = {word:i+1 for i,word in enumerate(vocab)}
	idx2word = {i+1:word for i,word in enumerate(vocab)}

	vec_texts = vectorize_review(word2idx, texts)

	lens = [len(text) for text in vec_texts]
	print(max(lens), min(lens), np.mean(lens))

	# print(vocab)
	return vec_texts, vocab, word2idx



def vectorize_review(word2idx, texts):
	vec_texts = [[0]]

	for text in texts:
		tokens = nltk.word_tokenize(text)
		vec_text = [word2idx[token] for token in tokens  if token in word2idx]
		vec_texts.append(vec_text)
	return vec_texts

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



def get_vocab_tfidf(corpus, vocab_size = 20000):
	from sklearn.feature_extraction.text import TfidfVectorizer
	tfidf_model = TfidfVectorizer().fit(corpus)
	sparse_result = tfidf_model.transform(corpus)
	mat = np.array(sparse_result.todense())
	print(mat.shape)
	vec = np.max(mat, axis=0)
	print(vec.shape)
	index = np.argsort(vec)[::-1]
	vocab = []
	# total_vocab_size = min(vocab_size, len(tfidf_model.vocabulary_.keys()))
	idx2word = {v:k for k,v in tfidf_model.vocabulary_.items()}

	for idx in index:
		word = idx2word[idx]
		if word not in stopwords:
			vocab.append(word)
			if len(vocab)>vocab_size:
				break
	return vocab


def validate(filename):
	filename = 'Musical_Instruments_5.json'
	data1 = readfile('./data/'+filename)
	data2 = pickle.load(open('./data/'+filename.split('.')[0]+'.pkl','rb'))


	length = len(data2['vec_uit'])
	sample = np.random.randint(length)
	uit = data2['vec_uit'][sample]
	print(uit)
	user_idx = uit[0]
	item_idx = uit[1]
	text_idx = uit[2]

	user2idx = data2['user2idx']
	item2idx = data2['item2idx']
	idx2user = {v[1]:v[0] for v in user2idx.items()}
	idx2item = {v[1]:v[0] for v in item2idx.items()}

	user = idx2user[user_idx]
	item = idx2item[item_idx]

	text = data1[text_idx-1]
	print(user, text['reviewerID'])
	print(item, text['asin'])

	u_text = data2['vec_u_text'][user_idx]
	i_text = data2['vec_i_text'][item_idx]

	print(u_text, i_text)

	for text_idx in u_text:
		text = data1[text_idx-1]
		print(text['reviewerID'])
	print(u_text)

def build_data(filename):
	data = readfile('./data/'+filename)
	# u_dict, i_dict = filter_ui(data, 5,5)
	# print(len(u_dict), len(i_dict))
	# data = data_after_filter(data, u_dict, i_dict)
	vec_texts, vocab, word2idx = build_vocab(data)
	vec_uit, vec_u_text, vec_i_text, user2idx, item2idx = build_ui_vocab(data)

	data_save = {}
	data_save['vec_texts'] 	= vec_texts
	data_save['vocab'] 		= vocab
	data_save['word2idx'] 	= word2idx
	data_save['vec_uit'] 	= vec_uit
	data_save['vec_u_text'] = vec_u_text
	data_save['vec_i_text'] = vec_i_text
	data_save['user2idx'] 	= user2idx
	data_save['item2idx'] 	= item2idx

	fr = open('./data/'+filename.split('.')[0]+'.pkl','wb')
	pickle.dump(data_save,fr)



if __name__ == '__main__':
	filename = 'Musical_Instruments_5.json'

	build_data(filename)
	# validate(filename)
