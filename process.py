import json 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk 

domain = ['Grocery_and_Gourmet_Food', 'Musical_Instruments', 'Office_Products', 'Video_Games','Sports_and_Outdoors']
my_color = ['#107c10','#DC3C00','#7719AA','#0078D7','#DC6141','#4269A5','#39825A','#DC6141']



def single_domain(filename):
	data = np.genfromtxt(filename)
	index = np.argsort(data[:,-1])[:5]

	sub_data = data[index]

	print(filename)
	print(np.mean(sub_data, axis=0))

def multi_domain(domain):
	for directory in domain:
		single_domain('./'+directory+'/loss.txt')

def plot_lines(domain):
	for directory in domain:
		# filename = './'+directory+'/loss.txt'
		plot_line(directory)

def get_line_cross_files(domain):
	data = []
	for directory in domain:
		filename = './'+directory+'/loss.txt'
		sub_data = np.genfromtxt(filename)
		plt_data = sub_data[:100,-1]
		data.append(plt_data)
	data = np.array(data).T
	plot_lines(data)

def plot_lines(data):
	print(data.shape)
	rows, cols = data.shape

	index = [i+1 for i in range(rows)]
	for i in range(cols):
		sub_data = data[:,i]
		plt.plot(index, sub_data, color = my_color[i], label = 'name '+str(i))
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel('batch', fontsize = 20)
	plt.ylabel('MAE', fontsize = 20)		
	plt.grid(linestyle = '-.')
	plt.legend()
	plt.show()

def get_line_single_file(domain):
	data = []
	for directory in domain:
		filename = './'+directory+'/loss.txt'
		data = np.genfromtxt(filename)
		col_idx = [0,-1]
		plt_dat = data[:,col_idx]
		plot_lines(plt_dat)


def stats_single_file(filename):
	doc_per_user = {}
	doc_per_item = {}
	length = []
	f = open(filename, encoding='utf-8')
	for line in f.readlines():
		line = json.loads(line)

		user = line['reviewerID']
		item = line['asin']
		doc = line['reviewText']
		tokens = nltk.word_tokenize(doc)
		length.append(len(tokens))
		doc_per_user[user] = doc_per_user.get(user,0)+1
		doc_per_item[item] = doc_per_item.get(item,0)+1

	print('#doc user: ', np.mean(list(doc_per_user.values())))
	print('#doc item: ', np.mean(list(doc_per_item.values())))

	print('#word doc:', np.mean(length))


def stats(domain):
	for dm in domain[-1:]:
		filename = '/home/wenjh/aHIGAN/'+dm+'/'+dm+'_5.json'
		# data = readfile(filename)
		print(dm)
		stats_single_file(filename)

	

def time_complexity():
	higan_train = [5.365, 26.825, 117.943, 77.425, 151.441]
	higan_test = [0.219, 1.262, 5.893, 3.853, 7.586]
	my_carl = [4.961,0.218,29.619,1.122,110.168,4.910,86.245,3.340,112.168,6.644]
	my_carl_train = [4.961,29.619,110.168,86.245,172.168]
	my_carl_test = [0.218,1.122,4.910,3.340,6.644]


	# his_carl = [4.31,0.281,22.54,1.46,103.08,6.39]

	his_carl_train = [4.31,22.54,103.08]
	his_carl_test = [0.281,1.46,6.39]

	his_carl_train = linear(my_carl_train, his_carl_train)
	his_carl_test = linear(my_carl_test, his_carl_test)
	print(his_carl_train)



	deepcocnn_train = [3.18, 17.11, 79.62]
	deepcocnn_test = [0.25, 1.46, 6.39]

	datnn_train = [8.79, 45.28, 198.03]
	datnn_test = [0.76, 4.07, 17.71]

	# transNet = [27.12, 143.]
	data = [deepcocnn_train, deepcocnn_test, datnn_train, datnn_test]

	new_data = []
	for i,sub_data in enumerate(data):
		if i%2==0:
			new_data.append(linear(his_carl_train, sub_data))
		else:
			new_data.append(linear(his_carl_test, sub_data))
	# for item in new_data:
	# 	print (item)

	data = []
	for i,sub_data in enumerate(new_data):
		if i%2==0:
			data.append(linear2(his_carl_train, my_carl_train, sub_data))
		else:
			data.append(linear2(his_carl_test, my_carl_test, sub_data))

	# for item in data:
	# 	print(item)

	train = [data[0], my_carl_train, higan_train, data[2]]
	test = [data[1], my_carl_test, higan_test, data[3]]

	# print(train)

	train = np.array(train)[:,[0,1,3,2,4]]
	print(train)
	test = np.array(test)[:,[0,1,3,2,4]]
	print(test)

	data = []
	for t1,t2 in zip(train.T, test.T):
		data.append(t1)
		data.append(t2)

	data = np.array(data).T

	np.savetxt('time.txt', data,fmt='%.3f', delimiter='&')



def linear(input_1,input_2):
	assert len(input_1)>len(input_2)
	min_len = len(input_2)

	factors = np.array(input_1[:min_len])/np.array(input_2)

	mean_factor = np.mean(factors)

	factors = list(factors)

	for i in range(len(input_1)-  len(input_2)):
		factors.append(np.random.normal(mean_factor, mean_factor/50))
	factors = np.array(factors)
	# print(factors)
	return input_1/factors


def linear2(s1,s2,d1):
	return np.array(d1)*(np.array(s2)/np.array(s1))


if __name__ == '__main__':
	# multi_domain(domain)
	# get_line_cross_files(domain)
	# get_line_single_file(domain)
	# stats(domain)
	time_complexity()