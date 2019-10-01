import json 
import numpy as np 
import matplotlib.pyplot as plt 
import nltk 

domain = ['Grocery_and_Gourmet_Food', 'Musical_Instruments', 'Office_Products', 'Video_Games']
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
	for dm in domain:
		filename = '/home/wenjh/aHIGAN/'+dm+'/'+dm+'_5.json'
		# data = readfile(filename)
		print(dm)
		stats_single_file(filename)

	

if __name__ == '__main__':
	# multi_domain(domain)
	# get_line_cross_files(domain)
	# get_line_single_file(domain)
	stats(domain)